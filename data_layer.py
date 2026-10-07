import json
import os
from typing import List, Dict, Any, Optional

class ProductRepository:
    def __init__(self, data_path: str = "data/products.json"):
        self.data_path = data_path
        self._products: List[Dict[str, Any]] = []
        self.load()

    def load(self):
        if os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as f:
                self._products = json.load(f)

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self._products)

    def get_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        for p in self._products:
            if p["product_id"].lower() == product_id.lower():
                return dict(p)
        return None


class OrderRepository:
    def __init__(self, data_path: str = "data/orders.json"):
        self.data_path = data_path
        self._orders: List[Dict[str, Any]] = []
        self.load()

    def load(self):
        if os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as f:
                self._orders = json.load(f)

    def get_by_id(self, order_id: str) -> Optional[Dict[str, Any]]:
        for o in self._orders:
            if o["order_id"].lower() == order_id.strip().lower():
                order = dict(o)
                # Compute derived total
                total = sum(item["quantity"] * item["price"] for item in order.get("items", []))
                order["total"] = total
                return order
        return None
