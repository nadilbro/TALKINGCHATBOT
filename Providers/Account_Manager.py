from SQL.SQLManager import VectorRAGService
from typing import Optional
import stripe
import os
from Providers.gemeni import DIAGRAM_PROMPT
 
# Pricing — all in USD per million tokens unless noted
COST_PER_1K_ELEVENLABS = 0.08              # USD per 1000 characters (Flash/Turbo)
COST_PER_MIN_DEEPGRAM = 0.0043             # USD per minute (Nova-3)
COST_PER_SEARCH_TAVILY = 0.008 * 10        # USD per search batch (avg 3 searches)
 
# Gemini 2.5 Pro — used when pro_mode=True
COST_PER_GEMINI_2PRO_INPUT_1M = 2.0        # $1.25 for <200k tokens, $2.50 for >200k — using avg
COST_PER_GEMINI_2PRO_OUTPUT_1M = 11.0      # $10 for <200k, $15 for >200k — using avg
 
# Gemini 2.5 Flash — default chat model
COST_PER_GEMINI_2FLASH_INPUT_1M = 0.30
COST_PER_GEMINI_2FLASH_OUTPUT_1M = 2.50
 
# Gemini 3 Flash Preview — diagram / code generation
COST_PER_GEMINI_3FLASH_INPUT_1M = 0.75
COST_PER_GEMINI_3FLASH_OUTPUT_1M = 4.50
 
# Gemini image input pricing
# Flash charges images at the same $0.30/1M text rate, tokenized per tile.
# A typical uploaded image is 1-8 tiles at 258 tokens each.
# We approximate a single image at 1500 tokens = roughly a 1200x1200px image.
# If you want exact billing, integrate Gemini's count_tokens API instead.
IMAGE_TOKENS_AVERAGE = 1500
 
# 1 token ≈ 4 characters of English
CHARS_PER_TOKEN = 4
 
# AUD conversion (rough — update periodically or pull from an API)
USD_TO_AUD = 1.55
 
 
class AccountManager:
 
    def __init__(self, rag: VectorRAGService):
        self.rag = rag
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
 
    ####################
    ## COST TRACKING  ##
    ####################
 
    def processUsedCost(
        self,
        outputText: str,
        outputDiagramText: str,
        inputText: str,
        SST_Length_seconds: float = 0,
        webSearch: bool = False,
        voice_on: bool = False,
        diagram_on: bool = False,
        pro_mode: bool = False,
        file_text_chars: int = 0,
        image_count: int = 0,
    ) -> float:
        """
        Returns the total cost of this turn in AUD.
 
        Parameters:
            outputText: The character's spoken response (billed as chat output)
            outputDiagramText: The SVG or code content (billed as diagram output)
            inputText: The user prompt + conversation history (billed as chat input)
            SST_Length_seconds: Audio length for Deepgram STT billing
            webSearch: True if Tavily was called
            voice_on: True if ElevenLabs TTS was used
            diagram_on: True if a diagram OR code was generated
            pro_mode: True if using Gemini 2.5 Pro instead of Flash
            file_text_chars: Number of characters of extracted file text added to
                             the prompt (for PDFs, DOCX, TXT, etc.). Pass 0 if no
                             text file was uploaded.
            image_count: Number of raw images sent to Gemini as vision input.
                         Pass 0 if no images were uploaded.
        """
        cost_usd = 0.0
 
        # ----------------------------------------------------------
        # Compute effective input character count
        # File text content adds to Gemini's input tokens even though it
        # lives in the system_prompt rather than inputText, so we fold
        # it in here.
        # ----------------------------------------------------------
        effective_input_chars = len(inputText) + file_text_chars
        chat_input_tokens = effective_input_chars / CHARS_PER_TOKEN
        chat_output_tokens = len(outputText) / CHARS_PER_TOKEN
 
        # ----------------------------------------------------------
        # CHAT CALL — Gemini 2.5 Flash or Pro depending on pro_mode
        # ----------------------------------------------------------
        if pro_mode:
            input_rate = COST_PER_GEMINI_2PRO_INPUT_1M
            output_rate = COST_PER_GEMINI_2PRO_OUTPUT_1M
        else:
            input_rate = COST_PER_GEMINI_2FLASH_INPUT_1M
            output_rate = COST_PER_GEMINI_2FLASH_OUTPUT_1M
 
        cost_usd += chat_input_tokens * input_rate / 1_000_000
        cost_usd += chat_output_tokens * output_rate / 1_000_000
 
        # ----------------------------------------------------------
        # IMAGE INPUT — only if actual images were uploaded (not text files)
        # Images are billed at the chat model's input rate, tokenized per tile
        # ----------------------------------------------------------
        if image_count > 0:
            image_input_tokens = image_count * IMAGE_TOKENS_AVERAGE
            cost_usd += image_input_tokens * input_rate / 1_000_000
 
        # ----------------------------------------------------------
        # DIAGRAM / CODE CALL — Gemini 3 Flash Preview (or Pro if pro_mode)
        # Only runs when the visual-aid router generated something
        # ----------------------------------------------------------
        if diagram_on:
            # Diagram input = the user's question + the full diagram system prompt
            diagram_input_chars = len(inputText) + len(DIAGRAM_PROMPT)
            diagram_output_chars = len(outputDiagramText)
 
            diagram_input_tokens = diagram_input_chars / CHARS_PER_TOKEN
            diagram_output_tokens = diagram_output_chars / CHARS_PER_TOKEN
 
            if pro_mode:
                cost_usd += diagram_input_tokens * COST_PER_GEMINI_2PRO_INPUT_1M / 1_000_000
                cost_usd += diagram_output_tokens * COST_PER_GEMINI_2PRO_OUTPUT_1M / 1_000_000
            else:
                cost_usd += diagram_input_tokens * COST_PER_GEMINI_3FLASH_INPUT_1M / 1_000_000
                cost_usd += diagram_output_tokens * COST_PER_GEMINI_3FLASH_OUTPUT_1M / 1_000_000
 
        # ----------------------------------------------------------
        # ELEVENLABS TTS — billed per 1000 characters of output text
        # ----------------------------------------------------------
        if voice_on:
            cost_usd += COST_PER_1K_ELEVENLABS * (len(outputText) / 1000)
 
        # ----------------------------------------------------------
        # DEEPGRAM STT — billed per minute of input audio
        # ----------------------------------------------------------
        if SST_Length_seconds > 0:
            cost_usd += COST_PER_MIN_DEEPGRAM * (SST_Length_seconds / 60)
 
        # ----------------------------------------------------------
        # TAVILY WEB SEARCH — flat rate per call
        # ----------------------------------------------------------
        if webSearch:
            cost_usd += COST_PER_SEARCH_TAVILY
 
        # ----------------------------------------------------------
        # Convert USD → AUD
        # ----------------------------------------------------------
        cost_aud = cost_usd * USD_TO_AUD
 
        print(
            f"==> COST DEBUG: inputText={len(inputText)} chars, "
            f"fileText={file_text_chars} chars, images={image_count}, "
            f"outputText={len(outputText)} chars, "
            f"outputDiagram={len(outputDiagramText)} chars, "
            f"voice={voice_on}, diagram={diagram_on}, pro={pro_mode}, "
            f"stt={SST_Length_seconds}s"
        )
        print(
            f"==> COST DEBUG: cost_usd=${cost_usd:.5f}, "
            f"cost_aud=${cost_aud:.5f}, credits={cost_aud / 0.15:.4f}"
        )
 
        return cost_aud
    ####################
    ## LIMIT CHECKING ##
    ####################

    def checkCostLimit(self, user_id: str, currentCost: Optional[float] = None) -> bool:
        """
        Returns True if user is within their monthly limit.
        Returns False if they have exceeded it.
        """
        if currentCost is None:
            currentCost = self.rag.getCurrentCost(user_id)
        limitCost = self.rag.getCostLimit(user_id)

        if limitCost is None:
            return True  # No limit set — allow through

        return currentCost < limitCost

    def checkAndResetBillingCycle(self, user_id: str) -> bool:
        """
        Checks if 30 days have passed since billing_cycle_start.
        If so, verifies Stripe subscription is still active, then resets usage.
        Returns True if reset happened, False otherwise.
        """
        cycle_expired = self.rag.checkBillingCycleExpired(user_id)

        if not cycle_expired:
            return False

        # Verify subscription is still active before resetting
        is_active = self.checkSubscription(user_id)
        if is_active:
            self.rag.resetMonthlyCost(user_id)
            return True
        else:
            # Subscription lapsed — don't reset, let limit block them
            return False

    #####################
    ## STRIPE          ##
    #####################

    def checkSubscription(self, user_id: str) -> bool:
        """
        Checks if the user has an active Stripe subscription.
        Returns True if active, False if cancelled/expired/none.
        """
        try:
            stripe_customer_id = self.rag.getStripeCustomerId(user_id)
            if not stripe_customer_id:
                return False

            subscriptions = stripe.Subscription.list(
                customer=stripe_customer_id,
                status="active",
                limit=1,
            )
            return len(subscriptions.data) > 0

        except Exception as e:
            print(f"Stripe check failed for {user_id}: {e}")
            return False

    def startSubscription(self, user_id: str, price_id: str) -> str:
        """
        Creates a Stripe checkout session for the user.
        Returns the checkout URL to redirect the user to.
        price_id: your Stripe price ID (from Stripe dashboard)
        """
        email = self.rag.getUserEmail(user_id)

        # Create or retrieve Stripe customer
        stripe_customer_id = self.rag.getStripeCustomerId(user_id)
        if not stripe_customer_id:
            customer = stripe.Customer.create(email=email)
            stripe_customer_id = customer.id
            self.rag.setStripeCustomerId(user_id, stripe_customer_id)

        session = stripe.checkout.Session.create(
            customer=stripe_customer_id,
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url=os.getenv("STRIPE_SUCCESS_URL", "https://yourapp.com/success"),
            cancel_url=os.getenv("STRIPE_CANCEL_URL", "https://yourapp.com/cancel"),
        )

        return session.url

    def cancelSubscription(self, user_id: str) -> bool:
        """
        Cancels the user's active Stripe subscription at period end.
        Returns True if cancelled successfully.
        """
        try:
            stripe_customer_id = self.rag.getStripeCustomerId(user_id)
            if not stripe_customer_id:
                return False

            subscriptions = stripe.Subscription.list(
                customer=stripe_customer_id,
                status="active",
                limit=1,
            )

            if not subscriptions.data:
                return False

            # Cancel at period end so they keep access until billing date
            stripe.Subscription.modify(
                subscriptions.data[0].id,
                cancel_at_period_end=True,
            )
            return True

        except Exception as e:
            print(f"Stripe cancel failed for {user_id}: {e}")
            return False

    def updateSubscription(self, user_id: str, new_price_id: str) -> bool:
        """
        Upgrades or downgrades the user's subscription to a new plan.
        Returns True if updated successfully.
        """
        try:
            stripe_customer_id = self.rag.getStripeCustomerId(user_id)
            if not stripe_customer_id:
                return False

            subscriptions = stripe.Subscription.list(
                customer=stripe_customer_id,
                status="active",
                limit=1,
            )

            if not subscriptions.data:
                return False

            sub = subscriptions.data[0]
            stripe.Subscription.modify(
                sub.id,
                items=[{"id": sub["items"].data[0].id, "price": new_price_id}],
            )
            return True

        except Exception as e:
            print(f"Stripe update failed for {user_id}: {e}")
            return False

    ##############################
    ## USER PREFERENCE MANAGEMENT##
    ##############################
    # FOR LATER