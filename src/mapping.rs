use dialoguer::{Select, theme::ColorfulTheme};

use crate::{
    error::{MgetError, Result},
    source::modelscope::ModelScopeSource,
};

const PREFIX_MAPPINGS: &[(&str, &str)] = &[
    ("meta-llama/", "LLM-Research/"),
    ("Qwen/", "qwen/"),
    ("mistralai/", "AI-ModelScope/"),
    ("google/", "AI-ModelScope/"),
];

pub async fn resolve_modelscope_id(
    input: &str,
    explicit_id: Option<&str>,
    interactive: bool,
    source: &ModelScopeSource,
) -> Result<String> {
    if let Some(id) = explicit_id {
        return Ok(id.to_string());
    }

    if let Some(mapped) = map_by_prefix(input) {
        return Ok(mapped);
    }

    let candidates = source.search_models(input).await?;
    match candidates.as_slice() {
        [single] => Ok(single.clone()),
        [] => Err(MgetError::ModelMappingNotFound(input.to_string())),
        many if interactive && is_tty() => {
            let selected = Select::with_theme(&ColorfulTheme::default())
                .with_prompt("Select ModelScope repository")
                .items(many)
                .default(0)
                .interact()?;
            Ok(many[selected].clone())
        }
        many => Err(MgetError::AmbiguousModelMapping {
            input: input.to_string(),
            candidates: many.join(", "),
        }),
    }
}

pub fn map_by_prefix(input: &str) -> Option<String> {
    PREFIX_MAPPINGS
        .iter()
        .find_map(|(from, to)| input.strip_prefix(from).map(|rest| format!("{to}{rest}")))
}

fn is_tty() -> bool {
    std::io::IsTerminal::is_terminal(&std::io::stdout())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_known_prefix() {
        assert_eq!(
            map_by_prefix("meta-llama/Meta-Llama-3-8B-Instruct").as_deref(),
            Some("LLM-Research/Meta-Llama-3-8B-Instruct")
        );
    }

    #[test]
    fn unknown_prefix_is_none() {
        assert!(map_by_prefix("unknown/model").is_none());
    }
}
