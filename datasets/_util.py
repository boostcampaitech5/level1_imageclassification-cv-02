from enum import Enum

class MaskLabels(int, Enum):
    """_summary_
    MASK = 0
    INCORRECT = 1
    NORMAL = 2
    """
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    """_summary_
    MALE, FEMALE의 라벨 
    MALE.value == 0
    FEMAEL.value == 1

    def from_str -> str type의 label을 0,1의 값으로 변경
    """
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        """_summary_

        Args:
            value (str): label (str)

        Raises:
            ValueError: label 맞는 value(str)이 들어오지 않으면

        Returns:
            int: label
        """
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    """_summary_
    YOUNG = 0
    MIDDEL = 1
    OLD = 2

    def from_number -> str type의 나이 숫자를 YOUNG, MIDDEL, OLD value로 변경
    """
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 29:
            return cls.YOUNG
        elif value < 58:
            return cls.MIDDLE
        else:
            return cls.OLD


FILE_NAME = {
    "mask1": MaskLabels.MASK,
    "mask2": MaskLabels.MASK,
    "mask3": MaskLabels.MASK,
    "mask4": MaskLabels.MASK,
    "mask5": MaskLabels.MASK,
    "incorrect_mask": MaskLabels.INCORRECT,
    "normal": MaskLabels.NORMAL
}
