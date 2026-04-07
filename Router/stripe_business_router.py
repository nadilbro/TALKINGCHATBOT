import os
import stripe
from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from SQL.SQLManager import VectorRAGService
from Providers.firebase_auth import verify_token

router = APIRouter(prefix="/embed-stripe", tags=["embed-stripe"])
rag = VectorRAGService()

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
EMBED_STRIPE_WEBHOOK_SECRET = os.getenv("EMBED_STRIPE_WEBHOOK_SECRET")

# ------------------------------------------------------------------
# Credit pack pricing — map Stripe price IDs to credit amounts
# ------------------------------------------------------------------
# Set these as env vars and create the matching products in Stripe dashboard.
# Pricing idea: businesses get cheaper per-credit rate than consumers.
CREDIT_PACKS = {
    os.getenv("STRIPE_BUSINESS_PRICE_STARTER", ""): 500,    # e.g. $25 → 500 credits
    os.getenv("STRIPE_BUSINESS_PRICE_GROWTH", ""): 1200,  # e.g. $50 → 1200 credits
    os.getenv("STRIPE_BUSINESS_PRICE_SCALE", ""): 3000,   # e.g. $100 → 3000 credits
}


# ------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------
class BusinessCheckoutRequest(BaseModel):
    price_id: str


# ------------------------------------------------------------------
# POST /embed-stripe/create-checkout-session
# ------------------------------------------------------------------
@router.post("/create-checkout-session")
async def create_business_checkout_session(
    req: BusinessCheckoutRequest,
    user=Depends(verify_token),
):
    """
    Create a Stripe checkout session for buying a business credit pack.
    Auth required — only logged-in developers can buy business credits.
    """
    user_id = user["uid"]

    if req.price_id not in CREDIT_PACKS:
        raise HTTPException(status_code=400, detail="Invalid price ID")

    try:
        stripe_customer_id = rag.getStripeCustomerId(user_id)

        if not stripe_customer_id:
            email = rag.getUserEmail(user_id)
            customer = stripe.Customer.create(email=email or "")
            stripe_customer_id = customer.id
            rag.setStripeCustomerId(user_id, stripe_customer_id)

        session = stripe.checkout.Session.create(
            customer=stripe_customer_id,
            client_reference_id=user_id,
            payment_method_types=["card"],
            line_items=[{"price": req.price_id, "quantity": 1}],
            mode="payment",  # one-time purchase, not subscription
            metadata={
                "type": "business_credits",  # critical — webhook routes on this
                "user_id": user_id,
                "price_id": req.price_id,
            },
            success_url=(os.getenv("STRIPE_SUCCESS_URL") or "...").strip(),
            cancel_url=(os.getenv("STRIPE_CANCEL_URL") or "...").strip(),
        )

        return {"url": session.url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# GET /embed-stripe/packs
# ------------------------------------------------------------------
@router.get("/packs")
async def list_credit_packs():
    """Returns the available business credit packs for the dashboard to display."""
    return {
        "packs": [
            {"price_id": pid, "credits": amount}
            for pid, amount in CREDIT_PACKS.items()
            if pid  # filter out empty env vars
        ]
    }


# ------------------------------------------------------------------
# POST /embed-stripe/webhook
# ------------------------------------------------------------------
@router.post("/webhook")
async def business_stripe_webhook(request: Request):
    """
    Separate webhook for business credit purchases.
    Use a different webhook endpoint and signing secret in Stripe dashboard
    so consumer and business webhooks don't interfere with each other.
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, EMBED_STRIPE_WEBHOOK_SECRET
        )
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid Stripe signature")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    event_type = event["type"]
    obj = event["data"]["object"]

    try:
        if event_type == "checkout.session.completed":
            metadata = obj.metadata or {}

            # Only handle business credit purchases — ignore consumer events
            if metadata.get("type") != "business_credits":
                print(f"==> embed-stripe: ignoring non-business event")
                return {"status": "ok"}

            user_id = metadata.get("user_id") or obj.client_reference_id
            price_id = metadata.get("price_id")

            if not user_id or not price_id:
                print(f"==> embed-stripe: missing user_id or price_id in metadata")
                return {"status": "ok"}

            credits_to_add = CREDIT_PACKS.get(price_id, 0)
            if credits_to_add <= 0:
                print(f"==> embed-stripe: unknown price_id {price_id}")
                return {"status": "ok"}

            rag.addBusinessCredits(user_id, credits_to_add)
            print(f"==> embed-stripe: +{credits_to_add} business credits for {user_id}")

    except Exception as e:
        print(f"==> embed-stripe webhook error for {event_type}: {e}")
        return {"status": "error", "detail": str(e)}

    return {"status": "ok"}