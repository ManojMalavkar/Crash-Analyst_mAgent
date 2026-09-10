# 02_PyDyna_Agent — LS-DYNA Solver Agent

AI agent for LS-DYNA model setup, keyword management, and solver configuration using the PyDyna SDK (`ansys.dyna.core`).

## Purpose

Helps CAE engineers:
- Set up LS-DYNA models programmatically via PyDyna API
- Look up keyword card syntax and parameters
- Select appropriate material models, contact types, and control settings
- Validate keyword files before solver submission
- Generate PyDyna code from natural language

## Architecture

```
User: "Create MAT_024 with yield stress 250 MPa"
  → search_keyword("MAT_024") → keyword_reference.jsonl
  → search_pydyna_api("material piecewise linear") → pydyna_docs.jsonl
  → LLM generates PyDyna code with correct parameters
```

## Knowledge Base

Two JSONL files:

### pydyna_docs.jsonl — PyDyna Python API

Key classes in `ansys.dyna.core`:

| Class | Purpose | Module |
|-------|---------|--------|
| DynaBase | Base for all analysis types | ansys.dyna.core.pre.dynabase |
| DynaSolution | Solution controls | ansys.dyna.core.pre.dynabase |
| DynaMech | Explicit mechanics | ansys.dyna.core.pre.dynamech |
| DynaEM | Electromagnetics | ansys.dyna.core.pre.dynaem |
| DynaThermal | Thermal analysis | ansys.dyna.core.pre.dynathermal |
| DynaAirbag | Airbag deployment | ansys.dyna.core.pre.dynaairbag |
| Deck | Keyword deck reader/writer | ansys.dyna.core.pre.model |

Material classes: MatRigid (020), MatElastic (001), MatPiecewiseLinearPlasticity (024), MatJohnsonCook (015), MatSpotweld (100)

Contact classes: ContactSurface, ContactFormingOneWaySurfaceToSurface

### keyword_reference.jsonl — LS-DYNA Keywords

Categories: Material (~50), Contact (~20), Section (~10), Control (~15), Database (~10), Boundary (~10), Load (~8), Define (~8), Initial (~5), Constrained (~5)

## Agent Tools (Day 10)

search_keyword, lookup_keyword, search_pydyna_api, get_material_model, get_contact_type, validate_keyword

## Build Pipeline

```bash
python bin/ingest_keywords.py --source <pydyna_docs_path>
python bin/build_vector_db.py --rebuild
python app_cli.py
```
