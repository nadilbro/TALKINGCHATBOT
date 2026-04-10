from fastapi import APIRouter, Query

from Providers.APIContracts import SessionBase, SessionCreate, SessionDelete, AccountCreate, toggleDiagramPro, changeModel
from SQL.SQLManager import VectorRAGService
from Providers.ai_provider import AIProvider
from Providers.startup_provider import StartUp
from Providers.Account_Manager import AccountManager
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, Form


router = APIRouter(prefix="/startup", tags=["startup"])

rag = VectorRAGService()
account = AccountManager(rag)
ai = AIProvider(rag)
startup = StartUp()
from Providers.firebase_auth import verify_token
from fastapi import Depends


    
@router.get("/initialise_sessions")
#This is used to initialise the sessions so upon the user login, the old sessions can load.
async def initialise_sessions(user_id: str = Query(...), user=Depends(verify_token)):
    return rag.get_session_history(user_id)


@router.post("/create_session")
async def create_session_route(data: SessionCreate, user=Depends(verify_token)):
    chat_id = rag.create_session(
        user_id=data.user_id,
        title=data.title,
        avatar_name=data.avatar_name
    )
    return {"chat_id": chat_id}

@router.post("/delete_session")
async def delete_session_route(data: SessionDelete, user=Depends(verify_token)):
    success = rag.delete_session(user_id=data.user_id, chat_id=data.id)
    return {"success": success}

@router.get("/get_current_usage")
async def get_usage_route(user_id: str = Query(...), user=Depends(verify_token)):
    credits_remaining = rag.getCreditsRemaining(user_id)
    current_cost = rag.getTokens(user_id)  # actual dollar cost spent this month
    return {
        "credits_remaining": credits_remaining,
        "current_cost": current_cost,
    }

@router.get("/check_subscription")
async def check_subscription(user_id: str = Query(...), user=Depends(verify_token)):
    cycle_expired = rag.checkBillingCycleReset(user_id)
    is_subscribed = rag.getSubscriptionStatus(user_id)

    if cycle_expired:
        if is_subscribed:
            rag.resetCredits(user_id, 50)
            rag.resetBillingCycle(user_id)
        else:
            rag.resetCredits(user_id, 0)
            rag.setSubscriptionActive(user_id, False)

    # Free weekly credit for non-subscribed users
    if not is_subscribed:
        rag.grantFreeDailyCredit(user_id)

    credits = rag.getCreditsRemaining(user_id)
    is_subscribed = rag.getSubscriptionStatus(user_id)

    return {
        "is_subscribed": is_subscribed,
        "credits_remaining": credits,
    }

@router.post("/create")
async def create_account(data: AccountCreate, user=Depends(verify_token)):
    """
    Called once after Firebase signup. Creates the account row with full details.
    Safe to call multiple times — won't overwrite existing data.
    """
    rag.createAccount(data.user_id, data.email, data.name, data.phone)
    return {"success": True}


@router.get("/me")
async def get_account(user_id: str = Query(...), user=Depends(verify_token)):
    """Returns the user's account details."""
    return rag.getAccount(user_id)

# -----------------------------------------------------------------------
# Diagram Enable
# -----------------------------------------------------------------------
@router.post("/toggle_diagram")
async def toggle_diagram(data: toggleDiagramPro, user=Depends(verify_token)):
    try:
        rag.toggle_diagram_usage(user["uid"], data.toggle)
        return {"ok": True, "diagram_use": data.toggle}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update setting: {e}")

@router.get("/check_diagram_use")
async def check_diagram_use(user=Depends(verify_token)):
    try:
        return {"diagram_use": rag.get_diagram_usage(user["uid"])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch setting: {e}")
    
# -----------------------------------------------------------------------
# Diagram Enable
# -----------------------------------------------------------------------
@router.post("/toggle_pro_mode")
async def toggle_pro_mode(data: toggleDiagramPro, user=Depends(verify_token)):
    try:
        rag.toggle_pro_usage(user["uid"], data.toggle)
        return {"ok": True, "pro_use": data.toggle}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update setting: {e}")

@router.get("/check_pro_use")
async def check_diagram_use(user=Depends(verify_token)):
    try:
        return {"pro_use": rag.get_pro_usage(user["uid"])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch setting: {e}")
    
# -----------------------------------------------------------------------
# Diagram Enable
# -----------------------------------------------------------------------
@router.post("/change_model")
async def change_model(data=changeModel, user=Depends(verify_token)):
    try:
        rag.set_model(user["uid"], data.model)
        return {"ok": True, "model_use": data.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update setting: {e}")

@router.get("/get_model")
async def get_model(user=Depends(verify_token)):
    try:
        return {"model_use": rag.get_model(user["uid"])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch setting: {e}")