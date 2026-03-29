import os
import stripe
from fastapi import APIRouter, Request, HTTPException, Query
from pydantic import BaseModel
from SQL.SQLManager import VectorRAGService

router = APIRouter(prefix="/stripe", tags=["stripe"])
rag = VectorRAGService()

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
CREDITS_PER_MONTH = 50


# ------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------
class CheckoutRequest(BaseModel):
    user_id: str
    price_id: str
    mode: str  # "subscription" or "payment"


# ------------------------------------------------------------------
# POST /stripe/create-checkout-session
# ------------------------------------------------------------------
@router.post("/create-checkout-session")
async def create_checkout_session(req: CheckoutRequest):
    try:
        stripe_customer_id = rag.getStripeCustomerId(req.user_id)

        # Create Stripe customer if they don't have one yet
        if not stripe_customer_id:
            email = rag.getUserEmail(req.user_id)
            customer = stripe.Customer.create(email=email or "")
            stripe_customer_id = customer.id
            rag.setStripeCustomerId(req.user_id, stripe_customer_id)

        session = stripe.checkout.Session.create(
            customer=stripe_customer_id,
            client_reference_id=req.user_id,  # critical — webhook uses this to identify user
            payment_method_types=["card"],
            line_items=[{"price": req.price_id, "quantity": 1}],
            mode=req.mode,
            
            success_url=(os.getenv("STRIPE_SUCCESS_URL") or "...").strip(),
            cancel_url=(os.getenv("STRIPE_CANCEL_URL") or "...").strip(),
        )

        return {"url": session.url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# GET /stripe/subscription-status?user_id=...
# ------------------------------------------------------------------
@router.get("/subscription-status")
async def subscription_status(user_id: str = Query(...)):
    try:
        stripe_customer_id = rag.getStripeCustomerId(user_id)
        credits_remaining = rag.getCreditsRemaining(user_id)

        if not stripe_customer_id:
            return {
                "plan": "free",
                "credits_remaining": credits_remaining or 0,
                "period_end": None,
            }

        subscriptions = stripe.Subscription.list(
            customer=stripe_customer_id,
            status="active",
            limit=1,
        )

        if not subscriptions.data:
            return {
                "plan": "free",
                "credits_remaining": credits_remaining or 0,
                "period_end": None,
            }

        sub = subscriptions.data[0]
        period_end = sub["current_period_end"]  # unix timestamp

        return {
            "plan": "basic",
            "credits_remaining": credits_remaining or 0,
            "period_end": period_end,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# POST /stripe/webhook
# ------------------------------------------------------------------
@router.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid Stripe signature")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    event_type = event["type"]
    data = event["data"]["object"]

    # User subscribes or buys a top-up
    if event_type == "checkout.session.completed":
        user_id = data.get("client_reference_id")
        stripe_customer_id = data.get("customer")
        mode = data.get("mode")

        if not user_id:
            return {"status": "ok"}

        rag.setStripeCustomerId(user_id, stripe_customer_id)

        if mode == "payment":
            rag.addCredits(user_id, 15)
            print(f"==> Top-up: +15 credits for {user_id}")

    # Monthly renewal — reset credits
    elif event_type == "invoice.paid":
        stripe_customer_id = data.get("customer")
        user_id = rag.getUserIdByStripeCustomerId(stripe_customer_id)
        if user_id:
            rag.resetCredits(user_id, CREDITS_PER_MONTH)
            rag.resetBillingCycle(user_id)
            print(f"==> Invoice paid: reset to {CREDITS_PER_MONTH} credits for {user_id}")

    # Subscription changed
    elif event_type == "customer.subscription.updated":
        stripe_customer_id = data.get("customer")
        user_id = rag.getUserIdByStripeCustomerId(stripe_customer_id)
        if user_id:
            is_active = data.get("status") == "active"
            rag.setSubscriptionActive(user_id, is_active)
            print(f"==> Subscription updated: {user_id} active={is_active}")

    # Subscription cancelled
    elif event_type == "customer.subscription.deleted":
        stripe_customer_id = data.get("customer")
        user_id = rag.getUserIdByStripeCustomerId(stripe_customer_id)
        if user_id:
            rag.setSubscriptionActive(user_id, False)
            rag.resetCredits(user_id, 0)
            print(f"==> Subscription cancelled: revoked {user_id}")

    return {"status": "ok"}