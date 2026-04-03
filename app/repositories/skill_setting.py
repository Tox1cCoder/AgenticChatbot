"""
Repository for managing per-user skill enable/disable settings.
"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.models.skill_setting import SkillSetting


class SkillSettingRepository:
    """Repository for SkillSetting model."""

    def __init__(self, session: Session | AsyncSession):
        self.session = session

    async def get_or_create(
        self,
        user_id: UUID,
        skill_name: str,
        default_enabled: bool = True,
    ) -> SkillSetting:
        """
        Get an existing skill setting or create if it doesn't exist.

        Args:
            user_id: The user ID.
            skill_name: The skill name.
            default_enabled: Default enabled state if creating new.

        Returns:
            The SkillSetting instance.
        """
        result = await self.session.execute(
            select(SkillSetting).where(
                SkillSetting.user_id == user_id,
                SkillSetting.skill_name == skill_name,
            )
        )
        setting = result.scalar_one_or_none()

        if setting is None:
            setting = SkillSetting(
                user_id=user_id,
                skill_name=skill_name,
                enabled=default_enabled,
            )
            self.session.add(setting)
            await self.session.commit()
            await self.session.refresh(setting)

        return setting

    async def get_by_user_and_skill(
        self,
        user_id: UUID,
        skill_name: str,
    ) -> SkillSetting | None:
        """
        Get a skill setting by user and skill name.

        Args:
            user_id: The user ID.
            skill_name: The skill name.

        Returns:
            The SkillSetting if found, None otherwise.
        """
        result = await self.session.execute(
            select(SkillSetting).where(
                SkillSetting.user_id == user_id,
                SkillSetting.skill_name == skill_name,
            )
        )
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        user_id: UUID,
        enabled_only: bool = False,
    ) -> list[SkillSetting]:
        """
        List all skill settings for a user.

        Args:
            user_id: The user ID.
            enabled_only: If True, only return enabled skills.

        Returns:
            List of SkillSetting instances.
        """
        query = select(SkillSetting).where(SkillSetting.user_id == user_id)

        if enabled_only:
            query = query.where(SkillSetting.enabled)

        result = await self.session.execute(query.order_by(SkillSetting.skill_name))
        return list(result.scalars().all())

    async def get_enabled_skills_map(self, user_id: UUID) -> dict[str, bool]:
        """
        Get a mapping of skill names to their enabled status.

        Args:
            user_id: The user ID.

        Returns:
            Dict mapping skill_name -> enabled.
        """
        settings = await self.list_by_user(user_id)
        return {s.skill_name: s.enabled for s in settings}

    async def set_enabled(
        self,
        user_id: UUID,
        skill_name: str,
        enabled: bool,
    ) -> SkillSetting:
        """
        Set the enabled status for a skill.

        Creates the setting if it doesn't exist.

        Args:
            user_id: The user ID.
            skill_name: The skill name.
            enabled: The enabled state.

        Returns:
            The updated SkillSetting.
        """
        setting = await self.get_or_create(user_id, skill_name, default_enabled=enabled)

        if setting.enabled != enabled:
            setting.enabled = enabled
            await self.session.commit()
            await self.session.refresh(setting)

        return setting

    async def bulk_set_enabled(
        self,
        user_id: UUID,
        skill_states: dict[str, bool],
    ) -> list[SkillSetting]:
        """
        Set enabled status for multiple skills at once.

        Args:
            user_id: The user ID.
            skill_states: Dict mapping skill_name -> enabled.

        Returns:
            List of updated SkillSetting instances.
        """
        results = []

        for skill_name, enabled in skill_states.items():
            setting = await self.set_enabled(user_id, skill_name, enabled)
            results.append(setting)

        return results

    async def delete(
        self,
        user_id: UUID,
        skill_name: str,
    ) -> bool:
        """
        Delete a skill setting.

        Args:
            user_id: The user ID.
            skill_name: The skill name.

        Returns:
            True if deleted, False if not found.
        """
        setting = await self.get_by_user_and_skill(user_id, skill_name)
        if not setting:
            return False

        await self.session.delete(setting)
        await self.session.commit()

        return True
