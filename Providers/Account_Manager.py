from SQL.SQLManager import VectorRAGService
from typing import Optional
import stripe
import os

COST_PER_1K_ELEVENLABS = 0.08        # Flash/Turbo
COST_PER_MIN_DEEPGRAM = 0.0043       # nova-3
COST_PER_SEARCH_TAVILY = 0.008 * 3   # Pay As You Go (avg 3 searches)
COST_PER_GEMINI_INPUT_1M = 0.25      # Per Million tokens
COST_PER_GEMINI_OUTPUT_1M = 1.50     # Per Million tokens


class AccountManager:

    def __init__(self, rag: VectorRAGService):
        self.rag = rag
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

    ####################
    ## COST TRACKING  ##
    ####################

    def processUsedCost(
        self,
        user_id: str,
        outputText: str,
        inputText: str,
        SST_Length_seconds: float = 0,
        webSearch: bool = False,
    ):
        """
        Calculates the API cost for one conversation turn and adds it to the user's monthly usage.
        SST_Length_seconds: length of the user's voice input in seconds (0 if text only).
        """
        cost = 0.0
        outputCharacters = len(outputText)
        inputCharacters = len(inputText)

        cost += inputCharacters * COST_PER_GEMINI_INPUT_1M / 1_000_000
        cost += outputCharacters * COST_PER_GEMINI_OUTPUT_1M / 1_000_000
        cost += COST_PER_1K_ELEVENLABS * (outputCharacters / 1000)
        cost += COST_PER_MIN_DEEPGRAM * (SST_Length_seconds / 60)

        if webSearch:
            cost += COST_PER_SEARCH_TAVILY

        self.rag.updateCurrentCost(user_id, cost)

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