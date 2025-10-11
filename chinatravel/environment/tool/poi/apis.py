import os
import json


class Poi:
    def __init__(self, base_path: str = "../../database/poi/", en_version=False):

        city_list = [
            "beijing",
            "shanghai",
            "nanjing",
            "suzhou",
            "hangzhou",
            "shenzhen",
            "chengdu",
            "wuhan",
            "guangzhou",
            "chongqing",
        ]
        curdir = os.path.dirname(os.path.realpath(__file__))
        data_path_list = [
            os.path.join(curdir, f"{base_path}/{city}/poi.json") for city in city_list
        ]
        self.data = {}
        for i, city in enumerate(city_list):
            self.data[city] = json.load(open(data_path_list[i], "r", encoding="utf-8"))
            city_data = {}
            for name_pos in self.data[city]:
                name = name_pos["name"]
                pos = name_pos["position"]
                city_data[name] = tuple(pos)
            self.data[city] = city_data
            # self.data[city] = [
            #     (x["name"], tuple(x["position"])) for x in self.data[city]
            # ]
        city_cn_list = [
            "åäº¬",
            "ä¸æµ·",
            "åäº¬",
            "èå·",
            "æ­å·",
            "æ·±å³",
            "æé½",
            "æ­¦æ±",
            "å¹¿å·",
            "éåº",
        ]
        for i, city in enumerate(city_list):
            self.data[city_cn_list[i]] = self.data.pop(city)
        self.city_cn_list = city_cn_list
        self.city_list = city_list

    def search(self, city: str, name: str):
        if city in self.city_list:
            city = self.city_cn_list[self.city_list.index(city)]
        city_data = self.data[city]
        try:
            return city_data[name]
        except KeyError:
            return f"No such point in the city. Check the point name: {name}."


def test():
    poi = Poi()
    while True:
        query = input("è¯·è¾å¥æ¥è¯¢çpoiåç§°ï¼")
        if query == "exit":
            return
        print(poi.search("åäº¬", query))


if __name__ == "__main__":
    test()
