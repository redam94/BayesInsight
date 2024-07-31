from pydantic import BaseModel, PositiveFloat, model_validator, field_validator, computed_field, Field, ValidationError, NaiveDatetime
import pandas as pd
import numpy as np
import pymc as pm

from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Union, Literal, Tuple, Set, Callable, Dict, Any, Iterable
import warnings
import json
import os


from foottraffic.awb_model.types import (
    Normilization, FunctionalForms,
    VarType, ModelType, Dim,
    MFF_ROW, INDEXCOL, Distribution, 
    PosDist, ContDist, Adstock)
from foottraffic.awb_model.constants import MFFCOLUMNS



class MetaData(BaseModel):
    allowed_geos: Optional[Set[str]] = None
    allowed_products: Optional[Set[str]] = None
    allowed_outlets: Optional[Set[str]] = None
    allowed_campaigns: Optional[Set[str]] = None
    allowed_creatives: Optional[Set[str]] = None
    necessary_variables: Optional[Set[str]] = None
    start_period: Optional[NaiveDatetime] = None
    end_period: Optional[NaiveDatetime] = None
    periodicity: Literal["Weekly", "Daily"] = "Weekly"
    row_ids: Tuple[INDEXCOL, ...] = Field(default = ("Period",), min_length=1, max_length=6)

    
    
    def check_values_contained(self, other_set: Set[str], col: str) -> bool:
        try:
            result = self.set_from_col(col).union(other_set) == other_set
        except ValueError:
            return False
        return result
    
    def set_from_col(self, col: str) -> Set[str]:
        if col == "Geography":
            return self.allowed_geos
        if col == "Product":
            return self.allowed_products
        if col == "Outlet":
            return self.allowed_outlets
        if col == "Campaign":
            return self.allowed_campaigns
        if col == "Creative":
            return self.allowed_creatives
        raise ValueError(f"{col} not valid column")
        
class MFF(BaseModel):
    data: List[MFF_ROW] = Field(repr=False)
    #periodicity: Literal["Weekly", "Daily"] = "Weekly"
    #row_ids: Tuple[INDEXCOL, ...] = Field(default = ("Geography", "Period"), max_length=6, min_length=1)
    metadata: Optional[MetaData] = Field(default = None, repr=False)

    @classmethod
    def from_mff_df(cls, df: pd.DataFrame, metadata: Optional[MetaData]=None)->'MFF':
        data = df.to_dict('records')
        return cls(data=data, metadata=None)
    
    def initialize_metadata(self) -> MetaData:
        if not self.metadata is None:
            return self.metadata
        df = self.as_df()
        allowed_geos = set(str(geo) for geo in df["Geography"].unique())
        allowed_products = set(str(product) for product in df["Product"].unique())
        allowed_outlets = set(str(outlet) for outlet in df['Outlet'].unique())
        allowed_campaigns = set(str(campaign) for campaign in df['Campaign'].unique())
        allowed_creatives = set(str(creative) for creative in df['Creative'].unique())
        self.metadata = MetaData(
            allowed_geos=allowed_geos, 
            allowed_products=allowed_products,
            allowed_outlets=allowed_outlets,
            allowed_campaigns=allowed_campaigns,
            allowed_creatives=allowed_creatives)
    
    @field_validator('data')
    @classmethod
    def data_validator(cls, v):
        errors = []
        
        for col_name in MFFCOLUMNS:
            try:
                if any(col_name not in set(col for col in row.keys()) for row in v):
                    raise ValueError(f"{col_name} not in dataset")
            except ValueError as e:
                errors.append(e)
        if errors:
            raise ValueError(*errors)
        
        return v
    
    @model_validator(mode='after')
    def check_date_alignment(self):
        if self.metadata is None:
            self.initialize_metadata()
        
        if self.metadata.periodicity == "Weekly":
            day_to_align = pd.to_datetime(self.data[0]['Period']).day_name()
            if not all(day_to_align == pd.to_datetime(row['Period']).day_name() for row in self.data):
                raise ValueError("Weekly data must be aligned to the same date")
        return self
    
    @model_validator(mode='after')
    def check_necessary_variables(self):
        if self.metadata is None:
            self.initialize_metadata()
            return self
        if self.metadata.necessary_variables is None:
            return self
        try:
            assert all(necessary_var in self.get_unique_varnames() for necessary_var in self.metadata.necessary_variables)
        except AssertionError:
            raise ValueError(f"{self.metadata.necessary_variables} must exist in the mff")
        return self
    
    @model_validator(mode='after')
    def check_metadata(self):
        if self.metadata is None:
            self.initialize_metadata()
            return self
        df = self.as_df()
        columns = ["Geography", "Product", "Outlet", "Campaign", "Creative"]
        unique_vals = {col: set(df[col].unique()) for col in columns}
        exceptions = []
        for col in columns:
            if self.metadata.set_from_col(col) is None:
                continue
            try:
                assert self.metadata.check_values_contained(unique_vals[col], col)
            except AssertionError:
                exceptions.append(ValueError(f"Only {self.metadata.set_from_col(col)} are allowed found {unique_vals[col] - self.metadata.set_from_col(col)}"))
        if exceptions:
            raise Exception(*exceptions)
        return self

    def factorize(self, col: str) -> Tuple[np.ndarray, pd.Index]:
        if col == 'Period':
            warnings.warn("Creating factor for each period ensure there are sufficient degrees of freedom!")
        df = self.analytic_dataframe()
        try:
            return pd.factorize(df[col])
        except KeyError:
            var_cols = [my_col for my_col in MFFCOLUMNS if my_col not in (["VariableName", "VariableValue", "Period"] + list(self.row_ids))]
            raise ValueError(f"{col} not found in column names. VariableName format is VariableName_{'_'.join(var_cols)}")
        
    def to_json(self, file: Union[str, Path]) -> None:
        with open(file, 'w') as f:
            f.write(self.model_dump_json())
        
    def to_bundle(self, folder: Union[str, Path]) -> None:
        folder = Path(folder)
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.as_df().to_csv(folder/"data.csv", index=False)
        
        with open(folder/"metadata.json", 'w') as f:
            f.write(self.model_dump_json(exclude=['data', '_info']))
    
    @classmethod
    def from_json(cls, file: Union[str, Path]) -> 'MFF':
        json_data = json.load(file)
        try:
            return cls(data=json_data['data'], metadata=json_data['metadata'])
        except ValidationError:
            raise ValidationError
    
    @classmethod
    def from_bundle(cls, folder: Union[str, Path]) -> 'MFF':
        folder = Path(folder)
        with open(folder/"metadata.json", 'r') as f:
            metadata = json.load(f)
        return cls.load_from_file(folder/'data.csv', **metadata)
    
    @classmethod
    def load_from_file(cls, file: Union[str, Path], metadata: Optional[MetaData]=None, **kwargs) -> 'MFF':
        """Loads MFF from csv file"""
        file_extention = str(file).split('.')[-1]
        try:
            assert file_extention == 'csv'
        except AssertionError:
            raise ValueError(f"Must provide a csv file not a {file_extention} type file")
       
        data = pd.read_csv(file).to_dict(orient='records')
        return cls(data=data, metadata=metadata)
    
    def as_df(self) -> pd.DataFrame:
        """Returns raw dataframe"""
        return pd.DataFrame(self.data)
    
    def analytic_dataframe(self, row_id: Optional[Union[List[str], str]]=None, aggfunc: Union[str, Callable[[pd.Series], float]]='sum') -> pd.DataFrame:
        """Returns Analytical Dataframe as a pandas dataframe object"""
        if isinstance(row_id, str):
            row_id = [row_id]
        if row_id is None:
            row_id = list(self.metadata.row_ids)
        if not 'Period' in row_id:
            warnings.warn("Period must be included in row_id adding Period to row_id")
            row_id = ['Period'] + row_id
        df = pd.DataFrame(self.data)[MFFCOLUMNS]
        df["Period"] = pd.to_datetime(df["Period"])
        columns = [col for col in MFFCOLUMNS if not col in row_id+["VariableValue", "VariableName"]]
        df = df.pivot_table(index=row_id, columns=["VariableName"]+columns, values="VariableValue", aggfunc=aggfunc)
        df.columns = ["_".join(col) for col in df.columns]
        return df.reset_index()
    
    def get_unique_varnames(self) -> list:
        """Returns list of all variablenames in model"""
        unique_var_names = list(pd.DataFrame(self.data)["VariableName"].unique())
        return unique_var_names
    
    def head(self)->pd.DataFrame:
        """Returns head of data"""
        return pd.DataFrame(self.data)[MFFCOLUMNS].head()
    
    @computed_field
    @property
    def _info(self) -> str:
        df = pd.DataFrame(self.data)
        df["Period"] = pd.to_datetime(df["Period"])
        unique_attr = {}
        for col in MFFCOLUMNS:
            if col == "Period" or col=='VariableValue':
                continue
            unique_attr[col] = {"# Unique": df[col].nunique(), "Value Counts": df[col].value_counts().to_dict()}
        unique_attr["Period"] = {
            "Periodicity": self.periodicity,
            "Date Range": (df["Period"].min().strftime("%Y-%m-%d"), df["Period"].max().strftime("%Y-%m-%d")),
            "# Unique": df['Period'].nunique(),
        }
        if self.metadata is None:
            self.initialize_metadata()
        if self.metadata.periodicity == 'Weekly':
            unique_attr["Period"] = unique_attr["Period"] | {"Aligned": df["Period"].dt.day_name().unique()[0]}
        return json.dumps(unique_attr)
    
    
    def print_info(self) -> None:
        def print_dict_recursive(item, depth=0):
            if isinstance(item, MFF):
                item = json.loads(item._info)
            if not isinstance(item, dict):
                print((depth)*"  "+str(item))
                return
            for key, my_item in item.items():
        
                if not isinstance(my_item, dict):
                    print((depth*"  ") + key + ":" + " "+ str(my_item))
                    continue
                print((depth*"  ") + key + ":")
                print_dict_recursive(my_item, depth=depth+1)
        return print_dict_recursive(self)

class Parameters(BaseModel):
    normalizationMethod: Normilization = Normilization.none
    functionalForm: FunctionalForms = FunctionalForms.linear
    alpha: PositiveFloat = 0.0
    beta: PositiveFloat = 0.0
    lag: int = 0
    decay: float = 1
    combineWith: Optional[List[str]] = None

class VariableDef(BaseModel):
    variableName: str
    transformParams: Parameters
    coeff: Optional[float] = None
    splitVariables: Optional[List[str]] = None
    #varType: VarType = VarType.none

    @classmethod
    def load_from_csv(cls, file: Union[str, Path]):
        file_extention = str(file).split('.')[-1]
        try:
            assert file_extention == 'csv'
        except AssertionError:
            raise ValueError(f"Must provide a csv file not a {file_extention} type file")

class ModelDef(BaseModel):
    variables: List[VariableDef]
    modelType: ModelType = ModelType.ols
    data: MFF
    groupingDims: Optional[List[Dim]] = None
    timeDim: Optional[Literal["Period"]] = None

    @model_validator(mode = 'after')
    def varify_variables_in_data(self):

        if not all(var.variableName in self.data.get_unique_varnames() for var in self.variables):
            raise ValueError("Data must contain all variables in the variable list")
        return self
        
    @model_validator(mode='after')
    def varify_necessary_attributes_for_model_type(self):

        if self.modelType == ModelType.ols or self.modelType == ModelType.timeseries:
            if not self.groupingDims is None:
                warnings.warn(f"Grouping dimentions are ignored if model is of type {self.modelType}. Try using FixedEffects or MixedEffects model.")
        if self.modelType == ModelType.ols:
            if not self.timeDim is None:
                warnings.warn(f"Time dimention is ignored if model is of type OLS. Try using a timeseries model.")
        if self.modelType == ModelType.timeseries:
            if self.timeDim is None:
                raise AttributeError("timeDim must be defined in order to use timeseries model")     
        if self.modelType == ModelType.mixedEffect or self.modelType == ModelType.fixedEffect:
            if self.groupingDims is None:
                raise AttributeError(f"Grouping dims must be defined for models of type {self.modelType}")
            
        return self


class Prior(BaseModel):
    dist: Union[Distribution, PosDist] = Field(default=Distribution.normal, validate_default=True)
    params: Dict[str, Union[float, List[Union[float, List]]]] = Field(default_factory=lambda:{'mu': 0.0, 'tau': 1.0})
    dims: Optional[Tuple[Dim, ...]] = None

    def get_pymc_prior(self, name: str):
        params = {key: value if isinstance(value, float) else np.array(value) for key, value in self.params.items()}
        return getattr(pm, self.dist)(name, **params, dims=self.dims)
    
class PositivePrior(Prior):
    
    @field_validator('dist')
    @classmethod
    def only_positive_dists(cls, v)->PosDist:
        try:
            return PosDist(v)
        except ValueError:
            raise ValueError(f"{v} is not a valid positive distribution {PosDist._member_names_}")
        
class MediaTransform(BaseModel):
    adstock_type: Optional[Adstock]
    

class VariableDetails(BaseModel):
    variable_name: str
    variable_type: VarType
    fixed_effect_coeff_prior: Prior = Field(default=Prior(), 
                                            description="""
                                            Prior for fixed effect. This can be multidimensional. 
                                            (i.e. if you set dims for Product you will have 1 fixed effect for each product. 
                                            You cannot have a fixed and random effect in the same dimension.)
                                            """)
    functional_form: FunctionalForms = FunctionalForms.linear
    random_effect_prior: Optional[PositivePrior] = Field(default=None, 
                                                         description="""
                                                         Prior for random effect. All random effects are added to the 
                                                         fixed_effect_coeff before the coefficient transform is applied
                                                         """)
    



    

    