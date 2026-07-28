# Streamlit Model Widget State Design

## Goal

Eliminate Streamlit's duplicate widget-default warning for the agent model controls while preserving backend-loaded values and user edits across reruns.

## Design

The backend snapshot remains the single initialization source. `_sync_model_config_form_state` writes the effective model values into `st.session_state` before the controls render. Widgets that use those keys must therefore omit their own `value=` or `index=` defaults and let Streamlit read the existing keyed state.

The change is limited to model controls already initialized by `_sync_model_config_form_state`. It does not alter payload construction, provider/model validation, or API behavior.

## Testing

Add a regression test around the rendered model controls that fails when a Session State-backed control also supplies a competing default. Run the focused model-control tests, then the relevant demo test suite and a minimal Streamlit warning reproduction.

