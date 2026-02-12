from elefant.config.load_config import ConfigBase


class WandbConfig(ConfigBase):
    enabled: bool = True
    entity: str | None = None  # WandB组织/用户名，如果为None则使用默认
    project: str = "elefant"
    exp_name: str = "policy_model"
    tags: list[str] = []
    run_id: str | None = None
