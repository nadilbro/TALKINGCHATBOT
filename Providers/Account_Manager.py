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
 

#Claude Sonnet 4.6
COST_PER_CLAUDE_SONNET_46_INPUT_1M = 3 
COST_PER_CLAUDE_SONNET_46_OUTPUT_1M = 15
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
        SST_Length_seconds: float = 0,
        webSearch: bool = False,
        voice_on: bool = False,
        diagram_on: bool = False,
        pro_mode: bool = False,
        image_count: int = 0,
        model: str = 'gemini',
        # Call 1 — real tokens
        input_tokens: int = 0,
        output_tokens: int = 0,
        # Call 2 — real tokens
        diagram_input_tokens: int = 0,
        diagram_output_tokens: int = 0,
        # TTS still needs char count
        outputText: str = "",
    ) -> float:

        cost_usd = 0.0

        # ----------------------------------------------------------
        # CALL 1 — Chat tokens (real if Anthropic, estimated if Gemini)
        # ----------------------------------------------------------
        if input_tokens > 0 or output_tokens > 0:
            # Real tokens from Anthropic __USAGE__
            chat_input_tokens = input_tokens
            chat_output_tokens = output_tokens

        if model == 'gemeni':
            input_rate = COST_PER_GEMINI_2PRO_INPUT_1M if pro_mode else COST_PER_GEMINI_2FLASH_INPUT_1M
            output_rate = COST_PER_GEMINI_2PRO_OUTPUT_1M if pro_mode else COST_PER_GEMINI_2FLASH_OUTPUT_1M
        elif model == 'anthropic':
            input_rate = COST_PER_CLAUDE_SONNET_46_INPUT_1M
            output_rate = COST_PER_CLAUDE_SONNET_46_OUTPUT_1M
        else:
            input_rate = COST_PER_GEMINI_2FLASH_INPUT_1M
            output_rate = COST_PER_GEMINI_2FLASH_OUTPUT_1M

        cost_usd += chat_input_tokens * input_rate / 1_000_000
        cost_usd += chat_output_tokens * output_rate / 1_000_000

        # ----------------------------------------------------------
        # IMAGE INPUT
        # ----------------------------------------------------------
        if image_count > 0:
            cost_usd += (image_count * IMAGE_TOKENS_AVERAGE) * input_rate / 1_000_000

        # ----------------------------------------------------------
        # CALL 2 — Diagram (always estimated, no usage object returned)
        # ----------------------------------------------------------
        if diagram_on:
            if diagram_on and (diagram_input_tokens > 0 or diagram_output_tokens > 0):
                cost_usd += diagram_input_tokens * input_rate / 1_000_000
                cost_usd += diagram_output_tokens * output_rate / 1_000_000

        # ----------------------------------------------------------
        # TTS, STT, Search — unchanged
        # ----------------------------------------------------------
        if voice_on:
            cost_usd += COST_PER_1K_ELEVENLABS * (len(outputText) / 1000)
        if SST_Length_seconds > 0:
            cost_usd += COST_PER_MIN_DEEPGRAM * (SST_Length_seconds / 60)
        if webSearch:
            cost_usd += COST_PER_SEARCH_TAVILY * 15 

        cost_aud = cost_usd * USD_TO_AUD
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