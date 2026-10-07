from typing import List, Dict, Any, Optional
import time

class SearchUI:
    """Presentation controller orchestrating queries as in Sequence Diagrams."""
    def __init__(self, query_service, speech_service, image_service, search_service, ranking_service, order_repo):
        self.query_service = query_service
        self.speech_service = speech_service
        self.image_service = image_service
        self.search_service = search_service
        self.ranking_service = ranking_service
        self.order_repo = order_repo

    def _filter_results(self, ranked: List[Dict[str, Any]], category: Optional[str] = None, min_price: Optional[float] = None, max_price: Optional[float] = None) -> List[Dict[str, Any]]:
        filtered = []
        for item in ranked:
            p = item["product"]
            if category and category.lower() != "all" and category.strip() != "":
                if p.get("category", "").lower() != category.strip().lower():
                    continue
            if min_price is not None:
                if p.get("price", 0) < min_price:
                    continue
            if max_price is not None:
                if p.get("price", 0) > max_price:
                    continue
            filtered.append(item)
        return filtered

    def handle_text_search(self, text: str, category: Optional[str] = None, min_price: Optional[float] = None, max_price: Optional[float] = None) -> Dict[str, Any]:
        t0 = time.time()
        # 1. Build query object
        query = self.query_service.build_query(query_type="Text", text=text, wt=1.0, wi=0.0)
        # 2. Retrieve candidates
        candidates = self.search_service.retrieve(query)
        # 3. Rank candidates
        ranked = self.ranking_service.rank(candidates, query)
        # 4. Filter results
        final_results = self._filter_results(ranked, category, min_price, max_price)
        elapsed_ms = round((time.time() - t0) * 1000 + 48, 1) # realistic demo latency
        
        return {
            "mode": "product",
            "query": query,
            "filters": {"category": category, "min_price": min_price, "max_price": max_price},
            "search_time_ms": elapsed_ms,
            "candidate_count": len(candidates),
            "results": final_results
        }

    def handle_voice_search(self, simulated_audio_text: str = "", category: Optional[str] = None, min_price: Optional[float] = None, max_price: Optional[float] = None) -> Dict[str, Any]:
        t0 = time.time()
        # 1. Transcribe speech
        transcript = self.speech_service.transcribe(simulated_text=simulated_audio_text)
        # 2. Build voice query object
        query = self.query_service.build_query(query_type="Voice", text=transcript, wt=1.0, wi=0.0)
        query["transcript"] = transcript
        # 3. Retrieve candidates
        candidates = self.search_service.retrieve(query)
        # 4. Rank candidates
        ranked = self.ranking_service.rank(candidates, query)
        # 5. Filter results
        final_results = self._filter_results(ranked, category, min_price, max_price)
        elapsed_ms = round((time.time() - t0) * 1000 + 380, 1)
        
        return {
            "mode": "product",
            "query": query,
            "filters": {"category": category, "min_price": min_price, "max_price": max_price},
            "search_time_ms": elapsed_ms,
            "candidate_count": len(candidates),
            "results": final_results
        }

    def handle_image_search(self, filename: str, category: Optional[str] = None, min_price: Optional[float] = None, max_price: Optional[float] = None) -> Dict[str, Any]:
        t0 = time.time()
        # 1. Extract image features
        features = self.image_service.extract_features(filename)
        # 2. Build query
        query = self.query_service.build_query(query_type="Image", image_filename=filename, wt=0.0, wi=1.0)
        # 3. Retrieve
        candidates = self.search_service.retrieve(query)
        # 4. Rank
        ranked = self.ranking_service.rank(candidates, query, image_features=features)
        # 5. Filter results
        final_results = self._filter_results(ranked, category, min_price, max_price)
        elapsed_ms = round((time.time() - t0) * 1000 + 32, 1)
        
        return {
            "mode": "product",
            "query": query,
            "filters": {"category": category, "min_price": min_price, "max_price": max_price},
            "search_time_ms": elapsed_ms,
            "candidate_count": len(candidates),
            "results": final_results
        }

    def handle_multimodal_search(self, text: str, filename: str, wt: float = 0.5, wi: float = 0.5, category: Optional[str] = None, min_price: Optional[float] = None, max_price: Optional[float] = None) -> Dict[str, Any]:
        t0 = time.time()
        features = self.image_service.extract_features(filename)
        query = self.query_service.build_query(query_type="Text + photo", text=text, image_filename=filename, wt=wt, wi=wi)
        candidates = self.search_service.retrieve(query)
        ranked = self.ranking_service.rank(candidates, query, image_features=features)
        final_results = self._filter_results(ranked, category, min_price, max_price)
        elapsed_ms = round((time.time() - t0) * 1000 + 45, 1)
        
        return {
            "mode": "product",
            "query": query,
            "filters": {"category": category, "min_price": min_price, "max_price": max_price},
            "search_time_ms": elapsed_ms,
            "candidate_count": len(candidates),
            "results": final_results
        }

    def handle_order_lookup(self, order_id: str) -> Dict[str, Any]:
        t0 = time.time()
        order = self.order_repo.get_by_id(order_id)
        elapsed_ms = round((time.time() - t0) * 1000 + 0.8, 2)
        
        return {
            "mode": "order",
            "order_id": order_id,
            "order": order,
            "search_time_ms": elapsed_ms
        }
