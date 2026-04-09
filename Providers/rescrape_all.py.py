"""
scripts/rescrape_all.py
 
Standalone script that runs on a schedule (daily at 12:00 UTC) via Render Cron.
Finds all active API keys with a configured website_url and re-scrapes them
to keep the RAG content fresh.
 
Deployment:
    Add this as a Render Cron Job (not a web service).
    Schedule: 0 12 * * *  (daily at 12:00 UTC)
    Start command: python scripts/rescrape_all.py
 
Environment variables needed (same as main backend):
    DATABASE_URL
    GEMINI_API_KEY
    (whatever else your VectorRAGService needs)
"""
 
import asyncio
import sys
import os
import traceback
from datetime import datetime, timezone
 
# Add project root to path so we can import from the same modules as the main app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from SQL.SQLManager import VectorRAGService
from website_scraper import scrape_and_ingest_website
 
 
async def rescrape_all_active_websites():
    """
    Loops through every active API key with a website_url set and re-scrapes
    the site. Logs results and updates last_scrape_at on success.
    """
    print(f"==> Daily rescrape started at {datetime.now(timezone.utc).isoformat()}")
    
    rag = VectorRAGService()
    
    # Fetch all API keys that need re-scraping
    try:
        keys_to_scrape = rag.getActiveKeysWithWebsites()
    except Exception as e:
        print(f"==> FATAL: Failed to fetch active keys: {e}")
        traceback.print_exc()
        return
    
    if not keys_to_scrape:
        print("==> No active keys with websites. Nothing to do.")
        return
    
    print(f"==> Found {len(keys_to_scrape)} active keys to rescrape")
    
    stats = {
        "total": len(keys_to_scrape),
        "succeeded": 0,
        "failed": 0,
        "total_chunks": 0,
    }
    
    for i, key_data in enumerate(keys_to_scrape, start=1):
        api_key = key_data.get("key")
        website_url = key_data.get("website_url")
        business_name = key_data.get("business_name") or "unknown"
        
        if not api_key or not website_url:
            continue
        
        print(f"==> [{i}/{len(keys_to_scrape)}] Scraping {business_name} ({website_url})")
        
        try:
            result = await scrape_and_ingest_website(
                api_key=api_key,
                website_url=website_url,
                rag=rag,
                replace_existing=True,
            )
            
            if result["success"]:
                stats["succeeded"] += 1
                stats["total_chunks"] += result["chunks_created"]
                
                # Update last_scrape_at on success
                try:
                    rag.updateApiKey(
                        key=api_key,
                        last_scrape_at=datetime.now(timezone.utc),
                    )
                except Exception as e:
                    print(f"==>   Failed to update last_scrape_at: {e}")
                
                print(f"==>   ✓ {result['pages_fetched']} pages, {result['chunks_created']} chunks")
            else:
                stats["failed"] += 1
                print(f"==>   ✗ Failed: {result.get('errors', [])}")
        
        except Exception as e:
            stats["failed"] += 1
            print(f"==>   ✗ Exception: {e}")
            traceback.print_exc()
        
        # Small pause between customers to avoid pile-up
        await asyncio.sleep(2)
    
    print(f"==> Daily rescrape complete: {stats['succeeded']}/{stats['total']} succeeded, {stats['total_chunks']} total chunks")
 
 
if __name__ == "__main__":
    asyncio.run(rescrape_all_active_websites())