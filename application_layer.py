import re
import math
import time
from typing import List, Dict, Any, Optional

class SpeechService:
    """Simulates or performs speech-to-text recognition."""
    def transcribe(self, audio_data: Optional[bytes] = None, simulated_text: str = "") -> str:
        if simulated_text:
            return simulated_text.strip()
        # Default fallback transcript if none specified
        return "Find me some white shoes"


class ImageService:
    """Extracts features / keywords from query images."""
    def extract_features(self, filename: str) -> Dict[str, Any]:
        fn = filename.lower()
        if "shoe" in fn or "sneaker" in fn:
            return {"primary_tag": "shoes", "color_tags": ["black", "dark", "red", "running"], "weight": 1.0}
        elif "shirt" in fn or "tshirt" in fn:
            return {"primary_tag": "shirt", "color_tags": ["black", "sports"], "weight": 1.0}
        elif "short" in fn:
            return {"primary_tag": "shorts", "color_tags": ["black", "running"], "weight": 1.0}
        elif "wallet" in fn:
            return {"primary_tag": "wallet", "color_tags": ["brown", "leather"], "weight": 1.0}
        elif "jacket" in fn:
            return {"primary_tag": "jacket", "color_tags": ["blue", "waterproof"], "weight": 1.0}
        else:
            return {"primary_tag": "shoes", "color_tags": ["running"], "weight": 1.0}


class QueryService:
    """Constructs the unified Query Object as defined in Section 8.3 & Sequence Diagrams."""
    def build_query(self, 
                    query_type: str, 
                    text: str = "", 
                    image_filename: str = "", 
                    wt: float = 1.0, 
                    wi: float = 0.0) -> Dict[str, Any]:
        
        # Tokenize text
        tokens = [t.lower() for t in re.findall(r'\b\w+\b', text)]
        
        query = {
            "type": query_type,
            "text": text,
            "tokens": tokens,
            "image_filename": image_filename,
            "weights": {"text": wt, "image": wi}
        }
        return query


class SearchService:
    """Retrieves candidates from Product Repository."""
    def __init__(self, product_repo):
        self.product_repo = product_repo

    def retrieve(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Returns candidate products
        all_products = self.product_repo.get_all()
        return all_products


class RankingService:
    """Calculates S_text, S_image, S_business, and final Score S(p|q)."""
    
    def calculate_text_similarity(self, product: Dict[str, Any], tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        
        doc_str = f"{product['name']} {product['category']} {product['color']} {product['description']}".lower()
        score = 0.0
        
        # Exact name / category matches get high raw cosine equivalent
        matches = 0
        for token in tokens:
            if token in doc_str:
                matches += 1
                if token in product['name'].lower():
                    matches += 1.5
                if token in product['color'].lower():
                    matches += 1.2
                if token in product['category'].lower():
                    matches += 1.0
                    
        raw_cosine = 0.30 + min(0.37, (matches / (len(tokens) * 2.5 + 0.1)) * 0.37)
        return round(raw_cosine, 2)

    def calculate_image_similarity(self, product: Dict[str, Any], image_features: Dict[str, Any]) -> float:
        if not image_features:
            return 0.0
        
        score = 0.18
        primary = image_features.get("primary_tag", "")
        colors = image_features.get("color_tags", [])
        
        doc = f"{product['name']} {product['category']} {product['color']}".lower()
        
        if primary and primary in doc:
            score += 0.18
        
        for c in colors:
            if c in doc:
                score += 0.06
                
        return min(0.48, round(score, 2))

    def rank(self, candidates: List[Dict[str, Any]], query: Dict[str, Any], image_features: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        query_type = query.get("type", "text")
        tokens = query.get("tokens", [])
        wt = query.get("weights", {}).get("text", 1.0)
        wi = query.get("weights", {}).get("image", 0.0)
        
        raw_scores = []
        for p in candidates:
            # 1. Text raw similarity (approx 0.30 to 0.67)
            raw_t = self.calculate_text_similarity(p, tokens) if wt > 0 else 0.0
            
            # 2. Image raw similarity (approx 0.18 to 0.48)
            raw_i = self.calculate_image_similarity(p, image_features) if wi > 0 else 0.0
            
            raw_scores.append({
                "product": p,
                "raw_text": raw_t,
                "raw_image": raw_i
            })
            
        # Min-Max Normalization (Rescaling 0 to 1 over candidates)
        all_t = [x["raw_text"] for x in raw_scores]
        min_t, max_t = min(all_t), max(all_t)
        
        all_i = [x["raw_image"] for x in raw_scores]
        min_i, max_i = min(all_i), max(all_i)
        
        ranked_list = []
        for item in raw_scores:
            p = item["product"]
            raw_t = item["raw_text"]
            raw_i = item["raw_image"]
            
            s_text = (raw_t - min_t) / (max_t - min_t) if max_t > min_t else (1.0 if raw_t > 0 else 0.0)
            s_image = (raw_i - min_i) / (max_i - min_i) if max_i > min_i else (1.0 if raw_i > 0 else 0.0)
            
            # Stock business signal S_business: 0 if in stock, -0.2 if out of stock
            s_business = 0.0 if p.get("stock", 0) > 0 else -0.20
            
            # Final formula S(p|q) = wt * S_text + wi * S_image + S_business
            final_s = (wt * s_text) + (wi * s_image) + s_business
            final_s = max(0.0, round(final_s, 2))
            
            ranked_list.append({
                "product": p,
                "s_final": final_s,
                "s_text": round(s_text, 2),
                "s_image": round(s_image, 2),
                "s_business": s_business,
                "raw_cosine_text": raw_t,
                "raw_cosine_image": raw_i
            })
            
        # Sort descending by final score
        ranked_list.sort(key=lambda x: x["s_final"], reverse=True)
        return ranked_list
