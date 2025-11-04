
import math
import os
import sys
import time
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


sys.path.append("./../../../")
project_root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from chinatravel.environment.tools import Transportation

from agent.base import BaseAgent


@dataclass
class Attraction:
    name: str
    type: str
    price: float
    opentime: str
    endtime: str
    recommend_min: float
    recommend_max: float


@dataclass
class Restaurant:
    name: str
    cuisine: str
    price: float
    opentime: str
    endtime: str
    recommendedfood: str


@dataclass
class Accommodation:
    name: str
    price: float
    bed_count: int





class TPCAgent(BaseAgent):
    """Heuristic travel planner powered by a local Qwen LLM for preference parsing."""

    CITY_DIR_MAP: Dict[str, str] = {
        "上海": "shanghai",
        "北京": "beijing",
        "深圳": "shenzhen",
        "广州": "guangzhou",
        "重庆": "chongqing",
        "苏州": "suzhou",
        "成都": "chengdu",
        "杭州": "hangzhou",
        "武汉": "wuhan",
        "南京": "nanjing",
    }

    BREAKFAST_SLOT: Tuple[int, int] = (8 * 60, 9 * 60)
    ATTRACTION_SLOTS: Tuple[Tuple[int, int], ...] = (
        (9 * 60, 11 * 60),
        (13 * 60 + 30, 15 * 60 + 15),
        (16 * 60, 18 * 60),
    )
    LUNCH_SLOT: Tuple[int, int] = (12 * 60, 13 * 60)
    DINNER_SLOT: Tuple[int, int] = (18 * 60 + 30, 19 * 60 + 30)

    def __init__(self, **kwargs):
        super().__init__(name="TPC", **kwargs)

        self.database_root = os.path.join(
            project_root_path, "environment", "database"
        )
        self.preference_prompt = self._build_preference_prompt()
        self._travel_cache: Dict[Tuple[str, str], Dict[str, object]] = {}
        self._city_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.transport_api = Transportation()
        self.reset()

    # ------------------------------------------------------------------
    # Abstract API implementation
    # ------------------------------------------------------------------
    def reset(self):
        self.parsed_preferences: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, query, prob_idx, oralce_translation=False):
        self.reset_clock()

        start_city = query.get("start_city")
        target_city = query.get("target_city")
        days = int(query.get("days", 3))
        people_number = int(query.get("people_number", 2))
        nl_request = query.get("nature_language", "")

        try:
            self.parsed_preferences = self._parse_preferences(
                start_city=start_city,
                target_city=target_city,
                days=days,
                people=people_number,
                request_text=nl_request,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.parsed_preferences = {
                "avoid_pois": set(),
                "must_visit": set(),
                "preferred_types": set(),
                "food_keywords": set(),
                "pace": "balanced",
                "budget": "medium",
                "notes": [f"LLM parsing failed: {exc}"],
                "must_not_do": [],
            }

        itinerary = self._build_itinerary(
            start_city=start_city,
            target_city=target_city,
            days=days,
            people=people_number,
        )

        success = len(itinerary) == days and all(day["activities"] for day in itinerary)

        result = {
            "people_number": people_number,
            "start_city": start_city,
            "target_city": target_city,
            "itinerary": itinerary,
            "elapsed_time(sec)": round(time.time() - self.start_clock, 3),
            "preference_summary": self._export_preferences(),
        }

        return success, result

    # ------------------------------------------------------------------
    # Preference parsing utilities
    # ------------------------------------------------------------------
    def _build_preference_prompt(self) -> List[Dict[str, str]]:
        system_prompt = (
            "You are an assistant that extracts actionable planning signals "
            "from Chinese travel requests. Respond in JSON only."
        )
        user_template = (
            "请阅读以下旅行请求，从中提取偏好和限制，以 JSON 返回。\n"
            "JSON 模式:\n"
            "{\n"
            "  \"avoid_pois\": [城市中需要避开的景点名称],\n"
            "  \"must_visit_pois\": [必须安排的景点或地标名称],\n"
            "  \"preferred_attraction_types\": [喜欢的景点类型或主题],\n"
            "  \"food_preferences\": [餐饮偏好，如菜系或特色菜],\n"
            "  \"budget_level\": \"low|medium|high\",\n"
            "  \"pace\": \"relaxed|balanced|packed\",\n"
            "  \"notes\": [其他重要要求],\n"
            "  \"must_not_do\": [明确禁止的事项]\n"
            "}\n"
            "缺失的信息请使用空列表或 null，保持客观，不要臆测。\n"
            "起点城市: {start_city}\n"
            "目的地城市: {target_city}\n"
            "出行天数: {days}\n"
            "出行人数: {people}\n"
            "需求原文: {request}\n"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template},
        ]

    def _parse_preferences(
            self,
            start_city: str,
            target_city: str,
            days: int,
            people: int,
            request_text: str,
    ) -> Dict[str, object]:
        user_message = self.preference_prompt[1]["content"].format(
            start_city=start_city,
            target_city=target_city,
            days=days,
            people=people,
            request=request_text,
        )
        messages = [self.preference_prompt[0], {"role": "user", "content": user_message}]
        response = self.backbone_llm(messages, json_mode=True)
        parsed = json.loads(response)
        return self._normalise_preferences(parsed)

    def _normalise_preferences(self, parsed: Dict[str, object]) -> Dict[str, object]:
        def _to_set(values: Iterable[str]) -> Set[str]:
            if not values:
                return set()
            result = set()
            for value in values:
                if not value:
                    continue
                result.add(str(value).strip())
            return result

        pace = str(parsed.get("pace", "balanced")).lower()
        if pace not in {"relaxed", "balanced", "packed"}:
            pace = "balanced"

        budget = str(parsed.get("budget_level", "medium")).lower()
        if budget not in {"low", "medium", "high"}:
            budget = "medium"

        preferences = {
            "avoid_pois": _to_set(parsed.get("avoid_pois", [])),
            "must_visit": _to_set(parsed.get("must_visit_pois", [])),
            "preferred_types": _to_set(parsed.get("preferred_attraction_types", [])),
            "food_keywords": {v.lower() for v in _to_set(parsed.get("food_preferences", []))},
            "pace": pace,
            "budget": budget,
            "notes": list(parsed.get("notes", [])) if parsed.get("notes") else [],
            "must_not_do": list(parsed.get("must_not_do", [])) if parsed.get("must_not_do") else [],
        }
        return preferences

    def _export_preferences(self) -> Dict[str, object]:
        exportable = {}
        for key, value in self.parsed_preferences.items():
            if isinstance(value, set):
                exportable[key] = sorted(value)
            else:
                exportable[key] = value
        return exportable

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------
    def _load_city_tables(self, city: str) -> Dict[str, pd.DataFrame]:
        # Defensive initialisation in case the agent instance is restored
        # from a checkpoint or instantiated without calling ``__init__``.
        # This ensures runtime invocations (e.g. during evaluation scripts)
        # do not fail with ``AttributeError`` when the cache is accessed.
        if not hasattr(self, "_city_cache"):
            self._city_cache = {}
        if not getattr(self, "database_root", None):
            self.database_root = os.path.join(
                project_root_path, "environment", "database"
            )
        if city in self._city_cache:
            return self._city_cache[city]

        city_key = self.CITY_DIR_MAP.get(city)
        if city_key is None:
            raise ValueError(f"Unsupported city: {city}")

        tables = {}
        tables["attractions"] = pd.read_csv(
            os.path.join(
                self.database_root, "attractions", city_key, "attractions.csv"
            )
        )
        tables["restaurants"] = pd.read_csv(
            os.path.join(
                self.database_root,
                "restaurants",
                city_key,
                f"restaurants_{city_key}.csv",
            )
        )
        tables["accommodations"] = pd.read_csv(
            os.path.join(
                self.database_root, "accommodations", city_key, "accommodations.csv"
            )
        )

        self._city_cache[city] = tables
        return tables

    def _build_attraction_pool(self, city: str) -> List[Attraction]:
        df = self._load_city_tables(city)["attractions"]
        attractions: List[Attraction] = []
        for _, row in df.iterrows():
            price = row.get("price", 0.0)
            price = float(price) if not pd.isna(price) else 0.0
            recommend_min = row.get("recommendmintime", 1.0)
            recommend_min = float(recommend_min) if not pd.isna(recommend_min) else 1.0
            recommend_max = row.get("recommendmaxtime", 2.0)
            recommend_max = float(recommend_max) if not pd.isna(recommend_max) else 2.0
            attractions.append(
                Attraction(
                    name=str(row.get("name", "未知景点")),
                    type=str(row.get("type", "综合")),
                    price=price,
                    opentime=str(row.get("opentime", "08:00")),
                    endtime=str(row.get("endtime", "22:00")),
                    recommend_min=recommend_min,
                    recommend_max=recommend_max,
                )
            )
        return attractions

    def _build_restaurant_pool(self, city: str) -> List[Restaurant]:
        df = self._load_city_tables(city)["restaurants"]
        restaurants: List[Restaurant] = []
        for _, row in df.iterrows():
            price = row.get("price", 120.0)
            price = float(price) if not pd.isna(price) else 120.0
            recommendedfood = row.get("recommendedfood", "")
            if pd.isna(recommendedfood):
                recommendedfood = ""
            restaurants.append(
                Restaurant(
                    name=str(row.get("name", "本地餐厅")),
                    cuisine=str(row.get("cuisine", "")),
                    price=price,
                    opentime=str(row.get("opentime", "10:00")),
                    endtime=str(row.get("endtime", "22:00")),
                    recommendedfood=str(recommendedfood),
                )
            )
        return restaurants

    def _build_accommodation_pool(self, city: str) -> List[Accommodation]:
        df = self._load_city_tables(city)["accommodations"]
        hotels: List[Accommodation] = []
        for _, row in df.iterrows():
            price = row.get("price", 600.0)
            price = float(price) if not pd.isna(price) else 600.0
            bed = row.get("numbed", 1)
            bed = int(bed) if not pd.isna(bed) else 1
            bed = max(1, bed)
            hotels.append(
                Accommodation(
                    name=str(row.get("name", "舒适酒店")),
                    price=price,
                    bed_count=bed,
                )
            )
        return hotels

    # ------------------------------------------------------------------
    # Inter-city transport helpers
    # ------------------------------------------------------------------
    def _select_intercity_transport(
            self, start_city: str, target_city: str, people: int
    ) -> Tuple[Optional[Dict[str, object]], Optional[int]]:
        if not start_city or not target_city or start_city == target_city:
            return None, None

        cache_key = (start_city, target_city)
        if cache_key in self._travel_cache:
            data = self._travel_cache[cache_key]
            return data["activity"], data["arrival_minutes"]

        activity, arrival_minutes = self._pick_flight(start_city, target_city, people)
        if activity is None:
            activity, arrival_minutes = self._pick_train(start_city, target_city, people)

        if activity is not None:
            self._travel_cache[cache_key] = {
                "activity": activity,
                "arrival_minutes": arrival_minutes,
            }


        return activity, arrival_minutes

    def _pick_flight(
            self, start_city: str, target_city: str, people: int
    ) -> Tuple[Optional[Dict[str, object]], Optional[int]]:
        file_path = os.path.join(
            self.database_root, "intercity_transport", "airplane.jsonl"
        )
        if not os.path.exists(file_path):
            return None, None

        matches = []
        with open(file_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                if start_city not in record.get("From", ""):
                    continue
                if target_city not in record.get("To", ""):
                    continue
                arrival_minutes, same_day = self._compute_arrival_minutes(
                    record["BeginTime"], record["Duration"]
                )
                if not same_day:
                    continue
                matches.append((record, arrival_minutes))

        if not matches:
            return None, None

        record, arrival_minutes = min(
            matches, key=lambda x: self._time_to_minutes(x[0]["BeginTime"])
        )
        activity = {
            "type": "airplane",
            "start": record["From"],
            "end": record["To"],
            "FlightID": record["FlightID"],
            "start_time": record["BeginTime"],
            "end_time": record["EndTime"],
            "price": float(record.get("Cost", 0.0)),
            "cost": round(float(record.get("Cost", 0.0)) * people, 2),
            "tickets": people,
        }
        return activity, arrival_minutes

    def _pick_train(
            self, start_city: str, target_city: str, people: int
    ) -> Tuple[Optional[Dict[str, object]], Optional[int]]:
        file_name = f"from_{start_city}_to_{target_city}.json"
        file_path = os.path.join(
            self.database_root,
            "intercity_transport",
            "train",
            file_name,
        )
        if not os.path.exists(file_path):
            return None, None

        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                options = json.load(fh)
        except json.JSONDecodeError:
            return None, None

        viable = []
        for record in options:
            arrival_minutes, same_day = self._compute_arrival_minutes(
                record["BeginTime"], record["Duration"]
            )
            if not same_day:
                continue
            viable.append((record, arrival_minutes))

        if not viable:
            return None, None

        record, arrival_minutes = min(
            viable, key=lambda x: self._time_to_minutes(x[0]["BeginTime"])
        )
        activity = {
            "type": "train",
            "start": record["From"],
            "end": record["To"],
            "TrainID": record["TrainID"],
            "start_time": record["BeginTime"],
            "end_time": record["EndTime"],
            "price": float(record.get("Cost", 0.0)),
            "cost": round(float(record.get("Cost", 0.0)) * people, 2),
            "tickets": people,
        }
        return activity, arrival_minutes

    def _compute_arrival_minutes(self, begin: str, duration_hours: float) -> Tuple[int, bool]:
        start_minutes = self._time_to_minutes(begin)
        duration_minutes = int(round(float(duration_hours) * 60))
        arrival_total = start_minutes + duration_minutes
        return arrival_total, arrival_total < 24 * 60

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------
    def _build_itinerary(
            self,
            start_city: str,
            target_city: str,
            days: int,
            people: int,
    ) -> List[Dict[str, object]]:
        attraction_pool = self._build_attraction_pool(target_city)
        restaurant_pool = self._build_restaurant_pool(target_city)
        hotels = self._build_accommodation_pool(target_city)

        preferred_hotel = self._choose_hotel(hotels, people)
        selected_restaurants = self._choose_restaurants(restaurant_pool, days)
        if not selected_restaurants:
            selected_restaurants = [
                Restaurant(
                    name=f"{target_city}精选餐厅",
                    cuisine="",
                    price=120.0,
                    opentime="11:00",
                    endtime="21:30",
                    recommendedfood="",
                    latitude=0.0,
                    longitude=0.0,
                )
            ]
        ranked_attractions = self._rank_attractions(attraction_pool)

        intercity_activity, arrival_minutes = self._select_intercity_transport(
            start_city, target_city, people
        )
        return_activity, _ = self._select_intercity_transport(
            target_city, start_city, people
        )

        itinerary: List[Dict[str, object]] = []
        used_attractions: Set[str] = set()
        current_city = start_city

        for day_idx in range(days):
            day_number = day_idx + 1
            day_plan = {"day": day_number, "activities": []}

            is_first_day = day_number == 1
            is_last_day = day_number == days

            if is_first_day and intercity_activity is not None:
                day_plan["activities"].append(intercity_activity)

            if is_first_day and arrival_minutes is not None:
                start_base = max(9 * 60, arrival_minutes + 45)
            else:
                start_base = 9 * 60

            return_activity_today: Optional[Dict[str, object]] = None
            departure_limit: Optional[int] = None
            include_accommodation = True

            if is_last_day and return_activity is not None:
                return_activity_today = dict(return_activity)
                departure_minutes = self._time_to_minutes(
                    return_activity_today.get("start_time", "21:30")
                )
                buffer_minutes = 60
                tentative_limit = departure_minutes - buffer_minutes
                if tentative_limit <= start_base:
                    tentative_limit = max(start_base + 30, departure_minutes - 30)
                latest_departure_window = departure_minutes - 15
                if latest_departure_window <= start_base:
                    departure_limit = start_base
                else:
                    departure_limit = min(
                        max(tentative_limit, self.BREAKFAST_SLOT[1]),
                        latest_departure_window,
                    )
                include_accommodation = False

            attractions_for_day = self._allocate_attractions_for_day(
                ranked_attractions,
                used_attractions,
                start_base,
                departure_limit,
            )

            meal_cycle_len = len(selected_restaurants)
            breakfast_rest = selected_restaurants[(day_idx * 3) % meal_cycle_len]
            lunch_rest = selected_restaurants[(day_idx * 3 + 1) % meal_cycle_len]
            dinner_rest = selected_restaurants[(day_idx * 3 + 2) % meal_cycle_len]

            self._append_daily_schedule(
                day_plan["activities"],
                attractions_for_day,
                breakfast_rest,
                lunch_rest,
                dinner_rest,
                preferred_hotel,
                people,
                day_start_minutes=start_base,
                day_end_limit=departure_limit,
                include_accommodation=include_accommodation,
            )

            if return_activity_today is not None:
                day_plan["activities"].append(return_activity_today)

            next_city_hint = target_city if current_city == start_city else start_city
            current_city = self._attach_transports_for_day(
                day_plan["activities"], current_city, next_city_hint, people
            )

            itinerary.append(day_plan)

        return itinerary

    def _rank_attractions(self, attractions: List[Attraction]) -> List[Attraction]:
        avoid = {name for name in self.parsed_preferences.get("avoid_pois", set())}
        preferred_types = {
            t.lower() for t in self.parsed_preferences.get("preferred_types", set())
        }
        must_keywords = [
            keyword for keyword in self.parsed_preferences.get("must_visit", set())
        ]

        def attraction_priority(item: Attraction) -> Tuple[int, float]:
            score = 0
            if preferred_types:
                if any(keyword in item.type.lower() for keyword in preferred_types):
                    score += 2
            if must_keywords:
                if any(keyword in item.name for keyword in must_keywords):
                    score += 3
            score += 1 / (1 + max(item.recommend_max, 1.0))
            return (-score, item.price)

        ranked = [attr for attr in attractions if attr.name not in avoid]
        if not ranked:
            ranked = list(attractions)
        ranked.sort(key=attraction_priority)

        must_items = []
        for keyword in must_keywords:
            for attr in ranked:
                if keyword in attr.name and attr not in must_items:
                    must_items.append(attr)
                    break

        remaining = [attr for attr in ranked if attr not in must_items]
        return must_items + remaining

    def _choose_restaurants(self, restaurants: List[Restaurant], days: int) -> List[Restaurant]:
        if not restaurants:
            return []
        keywords = self.parsed_preferences.get("food_keywords", set())
        if keywords:
            filtered = [
                rest
                for rest in restaurants
                if any(
                    keyword in rest.cuisine.lower() or keyword in rest.recommendedfood.lower()
                    for keyword in keywords
                )
            ]
            if filtered:
                restaurants = filtered
        restaurants.sort(key=lambda r: r.price)
        needed = max(3, min(len(restaurants), max(3, days * 3)))
        return restaurants[:needed]

    def _choose_hotel(self, hotels: List[Accommodation], people: int) -> Accommodation:
        if not hotels:
            return Accommodation(name="舒适酒店", price=600.0, bed_count=2)
        budget = self.parsed_preferences.get("budget", "medium")
        hotels_sorted = sorted(hotels, key=lambda h: h.price)
        if budget == "low":
            return hotels_sorted[0]
        if budget == "high":
            return hotels_sorted[-1]
        return hotels_sorted[len(hotels_sorted) // 2]

    def _allocate_attractions_for_day(
            self,
            ranked_attractions: List[Attraction],
            used_attractions: Set[str],
            day_start: int,
            day_end_limit: Optional[int],
    ) -> List[Tuple[Attraction, Tuple[int, int]]]:
        pace = self.parsed_preferences.get("pace", "balanced")
        if pace == "relaxed":
            slots_to_use = self.ATTRACTION_SLOTS[:2]
        elif pace == "packed":
            slots_to_use = self.ATTRACTION_SLOTS
        else:
            slots_to_use = self.ATTRACTION_SLOTS[:3]

        allocated: List[Tuple[Attraction, Tuple[int, int]]] = []
        for slot_start, slot_end in slots_to_use:
            adjusted_start = max(slot_start, day_start)
            if day_end_limit is not None and adjusted_start >= day_end_limit:
                continue
            if adjusted_start >= 21 * 60:
                continue
            window_end = slot_end
            if day_end_limit is not None:
                window_end = min(window_end, day_end_limit)
            if window_end - adjusted_start < 45:
                continue
            attraction = self._find_attraction_for_slot(
                ranked_attractions, used_attractions, adjusted_start, window_end
            )
            if attraction is None:
                continue
            used_attractions.add(attraction.name)
            allocated.append((attraction, (adjusted_start, window_end)))
        return allocated

    def _find_attraction_for_slot(
            self,
            ranked_attractions: List[Attraction],
            used_attractions: Set[str],
            slot_start: int,
            slot_end: int,
    ) -> Optional[Attraction]:
        for attraction in ranked_attractions:
            if attraction.name in used_attractions:
                continue
            open_minutes = self._time_to_minutes(attraction.opentime)
            close_minutes = self._time_to_minutes(attraction.endtime)
            if close_minutes == 0:
                close_minutes = 24 * 60
            if slot_end > close_minutes or slot_start < open_minutes:
                continue
            return attraction
        for attraction in ranked_attractions:
            if attraction.name in used_attractions:
                continue
            open_minutes = self._time_to_minutes(attraction.opentime)
            if slot_start < open_minutes:
                continue
            return attraction
        return None

    def _append_daily_schedule(
            self,
            activities: List[Dict[str, object]],
            attractions_with_slots: List[Tuple[Attraction, Tuple[int, int]]],
            breakfast_rest: Optional[Restaurant],
            lunch_rest: Optional[Restaurant],
            dinner_rest: Optional[Restaurant],
            hotel: Accommodation,
            people: int,
            day_start_minutes: int,
            day_end_limit: Optional[int],
            include_accommodation: bool,
    ) -> None:
        def within_limit(slot_start: int, slot_end: int) -> bool:
            if day_end_limit is None:
                return True
            if slot_start >= day_end_limit:
                return False
            effective_end = min(slot_end, day_end_limit)
            return effective_end > slot_start + 15

        if (
                breakfast_rest is not None
                and day_start_minutes <= self.BREAKFAST_SLOT[0]
                and within_limit(self.BREAKFAST_SLOT[0], self.BREAKFAST_SLOT[1])
        ):
            breakfast_end = (
                min(self.BREAKFAST_SLOT[1], day_end_limit)
                if day_end_limit is not None
                else self.BREAKFAST_SLOT[1]
            )
            if breakfast_end > self.BREAKFAST_SLOT[0]:
                breakfast_activity = self._build_meal_activity(
                    breakfast_rest,
                    people,
                    self.BREAKFAST_SLOT[0],
                    breakfast_end,
                    "breakfast",
                )
                if breakfast_activity is not None:
                    activities.append(breakfast_activity)

        lunch_inserted = False

        for attraction, (slot_start, slot_end) in attractions_with_slots:
            if (
                    not lunch_inserted
                    and lunch_rest is not None
                    and slot_start >= self.LUNCH_SLOT[0]
                    and within_limit(self.LUNCH_SLOT[0], self.LUNCH_SLOT[1])
            ):
                lunch_end = (
                    min(self.LUNCH_SLOT[1], day_end_limit)
                    if day_end_limit is not None
                    else self.LUNCH_SLOT[1]
                )
                if lunch_end > self.LUNCH_SLOT[0]:
                    lunch_activity = self._build_meal_activity(
                        lunch_rest,
                        people,
                        self.LUNCH_SLOT[0],
                        lunch_end,
                        "lunch",
                    )
                    if lunch_activity is not None:
                        activities.append(lunch_activity)
                    lunch_inserted = True

            start_time = max(slot_start, self._time_to_minutes(attraction.opentime))
            if day_end_limit is not None:
                slot_end = min(slot_end, day_end_limit)
            if slot_end - start_time < 45:
                continue
            duration_minutes = int(max(60, min(120, attraction.recommend_max * 60)))
            end_time = min(slot_end, start_time + duration_minutes)
            if end_time <= start_time:
                continue
            activity = {
                "type": "attraction",
                "position": attraction.name,
                "start_time": self._minutes_to_time(start_time),
                "end_time": self._minutes_to_time(end_time),
                "price": round(attraction.price, 2),
                "cost": round(attraction.price * people, 2),
                "tickets": people,
            }
            activities.append(activity)

        if (
                lunch_rest is not None
                and not lunch_inserted
                and within_limit(self.LUNCH_SLOT[0], self.LUNCH_SLOT[1])
        ):
            lunch_end = (
                min(self.LUNCH_SLOT[1], day_end_limit)
                if day_end_limit is not None
                else self.LUNCH_SLOT[1]
            )
            if lunch_end > self.LUNCH_SLOT[0]:
                lunch_activity = self._build_meal_activity(
                    lunch_rest, people, self.LUNCH_SLOT[0], lunch_end, "lunch"
                )
                if lunch_activity is not None:
                    activities.append(lunch_activity)

        if dinner_rest is not None and within_limit(self.DINNER_SLOT[0], self.DINNER_SLOT[1]):
            dinner_end = (
                min(self.DINNER_SLOT[1], day_end_limit)
                if day_end_limit is not None
                else self.DINNER_SLOT[1]
            )
            if dinner_end > self.DINNER_SLOT[0]:
                dinner_activity = self._build_meal_activity(
                    dinner_rest,
                    people,
                    self.DINNER_SLOT[0],
                    dinner_end,
                    "dinner",
                )
                if dinner_activity is not None:
                    activities.append(dinner_activity)

        if include_accommodation:
            rooms_needed = math.ceil(people / max(1, hotel.bed_count))
            activities.append(
                {
                    "type": "accommodation",
                    "position": hotel.name,
                    "start_time": "21:30",
                    "end_time": "23:00",
                    "price": round(hotel.price, 2),
                    "cost": round(hotel.price * rooms_needed, 2),
                    "room_type": hotel.bed_count,
                    "rooms": rooms_needed,
                }
            )

    def _attach_transports_for_day(
            self,
            activities: List[Dict[str, object]],
            initial_city: str,
            alternate_city: str,
            people: int,
    ) -> str:
        active_city = initial_city
        other_city = alternate_city
        last_location: Optional[str] = None
        last_end_time: Optional[str] = None
        for activity in activities:
            before_location = self._activity_location(activity, phase="before")
            activity.setdefault("transports", [])
            if (
                    before_location
                    and last_location
                    and before_location != last_location
                    and last_end_time is not None
            ):
                transports = self._plan_inner_transport(
                    active_city, last_location, before_location, last_end_time, people
                )
                activity["transports"] = transports
                if transports:
                    arrival_minutes = self._time_to_minutes(transports[-1]["end_time"])
                    start_minutes = self._time_to_minutes(activity["start_time"])
                    if arrival_minutes > start_minutes:
                        delta = arrival_minutes - start_minutes
                        new_start = arrival_minutes
                        new_end = self._time_to_minutes(activity["end_time"]) + delta
                        activity["start_time"] = self._minutes_to_time(new_start)
                        activity["end_time"] = self._minutes_to_time(new_end)
            else:
                activity["transports"] = []

            after_location = self._activity_location(activity, phase="after")
            if after_location:
                last_location = after_location
            elif before_location:
                last_location = before_location
            if activity.get("end_time"):
                last_end_time = activity["end_time"]
            if activity.get("type") in {"airplane", "train"}:
                active_city, other_city = other_city, active_city
        return active_city

    def _activity_location(self, activity: Dict[str, object], phase: str) -> Optional[str]:
        activity_type = activity.get("type")
        if activity_type in {"airplane", "train"}:
            if phase == "before":
                return activity.get("start")
            return activity.get("end")
        return activity.get("position")

    def _plan_inner_transport(
            self,
            city: str,
            origin: str,
            destination: str,
            departure_time: str,
            people: int,
    ) -> List[Dict[str, object]]:
        if not departure_time:
            return []
        if not origin or not destination or origin == destination:
            return []
        city_key = self.CITY_DIR_MAP.get(city, city)
        try:
            walk_option = self.transport_api.goto(city_key, origin, destination, departure_time, "walk")
        except Exception:
            walk_option = None

        walk_distance = None
        if isinstance(walk_option, list) and walk_option:
            walk_distance = walk_option[-1].get("distance")

        if walk_distance is not None and walk_distance <= 1.2:
            seg = walk_option[0]
            return [
                {
                    "start": seg.get("start", origin),
                    "end": seg.get("end", destination),
                    "mode": seg.get("mode", "walk"),
                    "type": seg.get("mode", "walk"),
                    "start_time": seg.get("start_time", departure_time),
                    "end_time": seg.get("end_time", departure_time),
                    "cost": 0.0,
                    "distance": round(seg.get("distance", 0.0), 2),
                    "price": 0.0,
                    "tickets": people,
                }
            ]

        try:
            taxi_option = self.transport_api.goto(city_key, origin, destination, departure_time, "taxi")
        except Exception:
            taxi_option = None

        if not isinstance(taxi_option, list) or not taxi_option:
            return []

        segment = taxi_option[0]
        cars_needed = max(1, math.ceil(people / 4))
        base_price = round(segment.get("cost", 0.0), 2)
        total_cost = round(base_price * cars_needed, 2)
        return [
            {
                "start": segment.get("start", origin),
                "end": segment.get("end", destination),
                "mode": "taxi",
                "type": "taxi",
                "start_time": segment.get("start_time", departure_time),
                "end_time": segment.get("end_time", departure_time),
                "cost": total_cost,
                "distance": round(segment.get("distance", 0.0), 2),
                "price": base_price,
                "cars": cars_needed,
                "tickets": cars_needed,
            }
        ]

    def _build_meal_activity(
            self,
            restaurant: Restaurant,
            people: int,
            slot_start: int,
            slot_end: int,
            meal_type: str,
    ) -> Optional[Dict[str, object]]:
        open_minutes = self._time_to_minutes(restaurant.opentime)
        close_minutes = self._time_to_minutes(restaurant.endtime)
        if close_minutes == 0:
            close_minutes = slot_end
        start_minutes = max(slot_start, open_minutes)
        end_minutes = min(slot_end, close_minutes)
        if end_minutes <= start_minutes:
            return None
        return {
            "type": meal_type,
            "position": restaurant.name,
            "start_time": self._minutes_to_time(start_minutes),
            "end_time": self._minutes_to_time(end_minutes),
            "price": round(restaurant.price, 2),
            "cost": round(restaurant.price * people, 2),
            "tickets": people,
        }

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _time_to_minutes(time_str: str) -> int:
        try:
            hour, minute = time_str.split(":")
            hour = int(hour)
            minute = int(minute)
        except Exception:
            hour, minute = 8, 0
        if hour >= 24:
            hour = 23
            minute = 59
        return hour * 60 + minute

    @staticmethod
    def _minutes_to_time(minutes: int) -> str:
        minutes = max(0, min(minutes, 23 * 60 + 59))
        hour = minutes // 60
        minute = minutes % 60
        return f"{hour:02d}:{minute:02d}"
