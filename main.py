"""
FastAPI application for mol_property microservice.

Provides HTTP endpoints for molecular property prediction.
"""

from typing import Dict, Any, Optional, List
import os

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from handler import predict_properties, predict_properties_batch

app = FastAPI(
    title="mol_property Microservice",
    description="Molecular property prediction service using ChemProp models",
    version="0.1.0"
)

# Request/Response Models
class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    message: str


class SMILESRequest(BaseModel):
    """Request model for single SMILES string processing."""
    smiles: str
    property: Optional[List[str]] = None


class PropertyResult(BaseModel):
    """Prediction result for a single property."""
    property: str
    status: str
    results: Optional[float] = None
    error: Optional[str] = None


class MultiSMILESResponse(BaseModel):
    """Response model for SMILES processing with multiple properties."""
    smiles: str
    status: str
    results: Dict[str, PropertyResult]
    error: Optional[str] = None


class BatchSMILESResponse(BaseModel):
    """Response model for batch SMILES processing."""
    filename: str
    requested_properties: str
    total_smiles: int
    results: List[MultiSMILESResponse]

# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        HealthResponse with service status
    """
    return HealthResponse(
        status="healthy",
        message="mol_property microservice is running"
    )


@app.post("/smi", response_model=MultiSMILESResponse)
async def predict_property(request: SMILESRequest) -> MultiSMILESResponse:
    """
    Predict molecular properties from a SMILES string.

    Args:
        request: SMILESRequest containing SMILES and optional list of properties

    Returns:
        MultiSMILESResponse with prediction results for each requested property.
        IMPORTANT: We only have one property for this, property is ignored and all properties are returned.

    Raises:
        HTTPException: For invalid input or processing errors
    """
    # Input validation
    if not request.smiles or request.smiles.strip() == "":
        raise HTTPException(status_code=400, detail="SMILES string cannot be empty")

    result = predict_properties(request.smiles, request.property)
    if result is None:
        status = "error"
        error_message = "Prediction failed for the given SMILES string"
        return MultiSMILESResponse(smiles=request.smiles, status=status, results={}, error=error_message)
    
    property_results = {}
    for prop, value in result.items():
        if type(value) != float:
            property_results[prop] = PropertyResult(
                property=prop,
                status="error",
                error="Prediction failed for this property"
            )
        else:
            property_results[prop] = PropertyResult(
                property=prop,
                status="success",
                results=value,
                error="None"
            )

    return MultiSMILESResponse(smiles=request.smiles, status="success",
                               results=property_results, error="None")


@app.post("/upload-smi", response_model=BatchSMILESResponse)
async def upload_smiles_file(file: UploadFile = File(...), property: Optional[List[str]] = None) -> BatchSMILESResponse:
    """
    Upload a file with a list of SMILES strings for batch prediction.

    The file should contain one SMILES string per line.

    Args:
        file: File containing SMILES strings (one per line)
        property: IGNORED

    Returns:
        BatchSMILESResponse with predictions for all SMILES

    Raises:
        HTTPException: For invalid file format or processing errors
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a name")

    try:
        # Read file contents
        contents = await file.read()
        text = contents.decode("utf-8")
        
        # Parse SMILES strings from file (one per line, skip empty lines)
        smiles_list = [line.strip() for line in text.split("\n") if line.strip()]
        
        if not smiles_list:
            raise HTTPException(status_code=400, detail="File contains no SMILES strings")
        
        # Process batch through handler
        results = predict_properties_batch(smiles_list, property)
        
        if results is None:
            error_message = "Batch prediction failed for the given file"
            return BatchSMILESResponse(
                filename=file.filename,
                requested_properties=property if property else "all",
                total_smiles=0,
                results=[],
                error=error_message
            )
        
        response_results = []
        for smi, res in zip(smiles_list, results):
            if res is not None:
                property_results = {}
                for prop, value in res.items():
                    property_results[prop] = PropertyResult(
                        property=prop,
                        status="success",
                        results=value,
                        error="None"
                    )
                response_results.append(MultiSMILESResponse(smiles=smi,
                                                            status="success",
                                                            results=property_results,
                                                            error="None"))
            else:
                property_result = PropertyResult(
                    property=property if property else "all",
                    status="error",
                    error="Prediction failed for this SMILES string"
                )
                response_results.append(MultiSMILESResponse(smiles=smi,
                                                            status="error",
                                                            results={property_result.property: property_result},
                                                            error=property_result.error))
        
        return BatchSMILESResponse(
            filename=file.filename,
            requested_properties=property if property else "all",
            total_smiles=len(smiles_list),
            results=response_results,
            error="None"
        )
    
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)