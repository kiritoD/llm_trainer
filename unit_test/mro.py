# 汽油车类
from dataclasses import dataclass

from attr import field


class GasolineCar(object):
    def run(self):
        print('i can run with gasoline!')

# 电动车类
class ElectricCar(object):
    def run(self):
        print('i can run with electric!')

# 混动汽车
class HybridCar(ElectricCar, GasolineCar):
    pass

# 了解一下以上代码的继承关系，到底HybridCar优先继承哪个类中的所有公共属性和公共方法呢？
print(HybridCar.__mro__)
print(HybridCar.mro())
a = HybridCar()
a.run()

@dataclass
class car:
    x: str = field(
        default="SIQADataset",
        metadata={"help": ("target dataset")},
    )
    y:int = 1

print(car.__dataclass_fields__['x'].default._default)