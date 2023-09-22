from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import importlib
from typing import Any, Dict, TypedDict
from types import ModuleType


class DataFrameEngineName(Enum):
    SPARK = "spark"
    PANDAS = "pandas"
    SNOWPARK = "snowpark"
    DUCKDB = "duckdb"
    POLARS = "polars"
    DATATABLE = "datatable"


class ModuleClassImport(TypedDict):
    module_name: str
    class_name: str


@dataclass
class DataFrame:
    dataframe_engine_name: DataFrameEngineName
    dataframe_engine: Any = None
    imports: Dict[str, ModuleClassImport] = field(default_factory=dict)

    def __new__(cls, *args: Any, **kwargs: Any) -> DataFrame:
        if not hasattr(cls, "instance"):
            cls.instance = super(DataFrame, cls).__new__(cls)
        return cls.instance

    def __init__(
        self,
        dataframe_engine_name: DataFrameEngineName,
        imports: Dict[str, ModuleClassImport] = {},
    ) -> None:
        default_imports: Dict[str, ModuleClassImport] = {
            "pandas": {
                "module_name": "pandas",
                "class_name": "DataFrame",
            },
            "spark": {
                "module_name": "pyspark.sql",
                "class_name": "DataFrame",
            },
            "snowpark": {
                "module_name": "snowflake.snowpark",
                "class_name": "DataFrame",
            },
            "duckdb": {
                "module_name": "duckdb",
                "class_name": "DuckDBPyConnection",
            },
            "polars": {
                "module_name": "polars",
                "class_name": "DataFrame",
            },
            "datatable": {
                "module_name": "datatable",
                "class_name": "Frame",
            },
        }

        self.imports: Dict[str, ModuleClassImport] = {**default_imports, **imports}
        self.dataframe_engine_name: DataFrameEngineName = dataframe_engine_name

        dataframe_module = importlib.import_module(
            self.imports[self.dataframe_engine_name.value]["module_name"]
        )
        dataframe_class: Any = getattr(
            dataframe_module,
            self.imports[self.dataframe_engine_name.value]["class_name"],
            None,
        )

        self.dataframe_engine: ModuleType = dataframe_class
