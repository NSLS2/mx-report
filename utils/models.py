from pydantic import BaseModel, validator, Field, root_validator
from uuid import UUID
from datetime import datetime
from typing import Optional, Literal, Union
from enum import Enum
from pathlib import Path


class RequestType(str, Enum):
    standard = "standard"
    raster = "raster"


class BaseRequestDefinition(BaseModel):
    sample: UUID
    sweep_start: float
    sweep_end: float
    img_width: float
    exposure_time: float
    protocol: RequestType
    detector_distance: float = Field(..., alias="detDist")
    parent_request_id: UUID | Literal[-1]
    base_path: Path
    file_prefix: str
    directory: Path
    file_number_start: int
    energy: float
    wavelength: float
    resolution: float
    slit_height: float
    slit_width: float
    attenuation: float
    visit_name: str
    detector: str
    beamline: str
    pos_x: float
    pos_y: float
    pos_z: float
    pos_type: str
    grid_step: float = Field(..., alias="gridStep")
    run_num: int = Field(..., alias="runNum")
    xbeam: float
    ybeam: float

    class Config:
        extra = "allow"  # Allow extra fields to pass into the model

    @root_validator(pre=True)
    def ignore_specific_fields(cls, values):
        # Removing dataPath because it is always stuck in /legacy/2022-3 which is of no use
        fields_to_ignore = {"dataPath"}
        return {k: v for k, v in values.items() if k not in fields_to_ignore}

    # Validator for fields of type `Path`
    @validator("base_path", "directory", pre=True, always=True)
    def convert_to_path(cls, value):
        return Path(value) if value else value


class StandardRequestDefinition(BaseRequestDefinition):
    centering_option: Literal["AutoRaster", "Manual"] = Field(
        ..., alias="centeringOption"
    )
    fast_dp: bool
    fast_ep: bool
    dimple: bool
    xia2: bool


class Point2D(BaseModel):
    x: float
    y: float


class RasterRowDefinition(BaseModel):
    start: Point2D
    end: Point2D
    num_steps: int = Field(..., alias="numsteps")


class RasterDefinition(BaseModel):
    beam_width: float = Field(..., alias="beamWidth")
    beam_height: float = Field(..., alias="beamHeight")
    status: int
    x: float
    y: float
    z: float
    omega: float
    step_size: float = Field(..., alias="stepsize")
    row_defs: list[RasterRowDefinition] = Field(..., alias="rowDefs")
    num_cells: int = Field(..., alias="numCells")
    raster_type: str = Field(..., alias="rasterType")


class MaxRasterPosition(BaseModel):
    file: str
    coords: list[float]
    index: int
    omega: float


class RasterRequestDefinition(BaseRequestDefinition):
    raster_def: RasterDefinition
    max_raster: MaxRasterPosition


RequestDefinition = Union[StandardRequestDefinition, RasterRequestDefinition]


class Request(BaseModel):
    uid: UUID
    sample: UUID
    time: float
    request_time: Optional[datetime] = None
    state: Optional[Literal["active", "inactive"]] = None
    seq_num: int
    priority: int
    request_type: RequestType
    request_def: RequestDefinition
    owner: str
    proposal_id: str = Field(..., alias="proposalID")

    @validator("request_time", always=True)
    def convert_epoch_to_datetime(cls, v, values):
        epoch_time = values.get("time")
        if epoch_time is not None:
            # Convert float epoch time to datetime
            return datetime.fromtimestamp(epoch_time)
        return v
