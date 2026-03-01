"""
Handler module for molecular property prediction.

Extracts molecular properties from SMILES strings using mol_property functions.
"""

from typing import Dict, Any, Optional, List
from rdkit import Chem
from mol_property import property_api


def predict_properties(smiles: str, requested_properties: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """
    Predict molecular properties for a given SMILES string.

    Args:
        smiles: SMILES string to process
        requested_properties: List of property names to extract. If None or empty, all properties are returned.

    Returns:
        Dictionary with predicted properties, or None if invalid SMILES
    """
    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Define all available properties
    all_properties = {
        "logP": property_api.get_logP(mol),
        "HBA": property_api.get_numHBA(mol),
        "HBD": property_api.get_numHBD(mol),
        "TPSA": property_api.get_polarSurfaceArea(mol),
        "RotatableBonds": property_api.get_numRotatableBonds(mol),
        "pKa": property_api.get_pKa(mol),
        "logS": property_api.get_logS(mol),
        "numRings": property_api.get_numRings(mol),
        "ruleOfFive": property_api.get_ruleOfFive(mol),
        "veberRule": property_api.get_veberRule(mol),
        "chemicalFormula": property_api.get_chemicalFormula(mol),
        "molecularMass": property_api.get_molecularMass(mol),
        "smiles": property_api.get_SMILES(mol),
    }
    
    try:
        # If no properties requested, return all
        if not requested_properties:
            return all_properties
        
        # Filter to requested properties only
        properties = {
            prop: all_properties[prop] 
            for prop in requested_properties 
            if prop in all_properties
        }
        return properties
    except Exception as e:
        print(f"Error extracting properties: {str(e)}")
        return None


def predict_properties_batch(smiles_list: List[str], requested_properties: Optional[List[str]] = None) -> List[Optional[Dict[str, Any]]]:
    """
    Predict molecular properties for a batch of SMILES strings.

    Args:
        smiles_list: List of SMILES strings to process
        requested_properties: List of property names to extract. If None or empty, all properties are returned.

    Returns:
        List of property dictionaries (None for invalid SMILES)
    """
    return [predict_properties(smiles, requested_properties) for smiles in smiles_list]
