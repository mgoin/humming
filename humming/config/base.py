import dataclasses
import enum
import re
from typing import Any, ClassVar

from typing_extensions import Self


def name_to_google_cpp_const_style(name: str) -> str:
    if not name:
        return ""
    name = name.strip().lower()
    words = re.split(r"[_ \W]+", name)
    pascal_words = [word.capitalize() for word in words if word]
    return "k" + "".join(pascal_words)


def name_value_to_google_cpp_const_style(name: str, value: Any, keep_name: bool = False) -> str:
    if not keep_name:
        name = name_to_google_cpp_const_style(name)
    if isinstance(value, bool):
        value = "true" if value else "false"
    elif isinstance(value, float):
        value = str(value) + "f"
    elif isinstance(value, int):
        value = str(value) + "u"
    else:
        value = str(value).replace(".", "::")

    return f"static constexpr auto {name} = {value};"


def name_value_to_extern_const_style(name: str, value: Any) -> str:
    name = name.upper()
    if isinstance(value, (bool, int)):
        value = int(value)
        return f'extern "C" __constant__ uint32_t {name} = {value};'
    return ""


@dataclasses.dataclass
class BaseHummingConfig:
    _name_map: ClassVar[dict[str, str]] = {}
    _cpp_ignore_names: ClassVar[tuple[str, ...]] = ()
    _cpp_extra_names: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self):
        for field in dataclasses.fields(self):
            name = field.name
            field_type = field.type
            value = getattr(self, name)
            msg = f"NAME: {name} ; FIELDTYPE: {field_type}; VALUE: {value}"

            if "typing.Optional" in str(field_type) or " | None" in str(field_type):
                if value is None:
                    continue
                field_type = field_type.__args__[0]
            if field_type is int:
                assert isinstance(value, int), msg
                assert value >= 0, msg
            elif field_type is float:
                assert isinstance(value, (int, float))
                value = float(value)
            elif field_type is bool:
                assert isinstance(value, bool), msg
            elif field_type is str:
                assert isinstance(value, str), msg
            elif isinstance(field_type, enum.EnumMeta):
                if isinstance(value, str):
                    value = getattr(field_type, value.upper())
                    setattr(self, name, value)
                else:
                    assert isinstance(value, field_type), msg
            else:
                raise ValueError("Invalid Field Type. " + msg)

    def to_cpp_str(self, include_class_name=False):
        str_list = []
        names = [x.name for x in dataclasses.fields(self)]
        names += list(self._cpp_extra_names)
        for name in names:
            if name in self._cpp_ignore_names:
                continue
            value = getattr(self, name)
            keep_name = name in self._name_map
            if keep_name:
                name = self._name_map[name]
            line = name_value_to_google_cpp_const_style(name, value, keep_name)
            str_list.append(line)

        code = "\n".join("  " + x for x in str_list)
        class_name = self.__class__.__name__

        if include_class_name:
            code = f"class {class_name} {{\n{code}\n}};"

        return code

    def to_extern_cpp_str(self):
        str_list = []
        names = [x.name for x in dataclasses.fields(self)]
        names += list(self._cpp_extra_names)
        for name in names:
            if name in self._cpp_ignore_names:
                continue
            value = getattr(self, name)
            line = name_value_to_extern_const_style(name, value)
            str_list.append(line)

        str_list = [x for x in str_list if x]
        code = "\n".join(x for x in str_list if x)

        return code

    @classmethod
    def from_dict(cls, raw_config: dict[str, Any]) -> Self:
        clean_config = cls._preprocess_dict(raw_config)
        return cls(**clean_config)

    @classmethod
    def _preprocess_dict(cls, config: dict[str, Any]) -> dict[str, Any]:
        if isinstance(config, BaseHummingConfig):
            config = config.__dict__
        assert isinstance(config, dict)
        field_name_list = set(x.name for x in dataclasses.fields(cls))
        return dict(item for item in config.items() if item[0] in field_name_list)
