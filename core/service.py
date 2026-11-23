import asyncio
import base64
import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import aiohttp
from aiocqhttp import CQHttp

from astrbot.api import logger

from .config import PluginConfig
from .draw import CardMaker
from .profile import (
    AVATAR_TIME_KEYS,
    BIRTHDAY_KEYS,
    COLLEGE_KEYS,
    EMAIL_KEYS,
    REG_TIME_KEYS,
    BoxUserProfile,
    flatten_info,
    friendly_duration,
    join_duration_suffix,
)
from .store import MemberStore

# 入群排位的「名册宽限窗口」：适配器成员列表可能滞后，事件里见过但列表还没收录的
# 新成员在此窗口内仍算在群；更早的记录视为幽灵（退群事件漏收），不计入总数。
PRESENCE_GRACE_SECONDS = 3600

# 注册时间/头像修改时间/生日等「时间档案」字段在 QQ 官方接口里都没有，只能指望
# 适配器扩展字段。首次出卡打一次诊断清单（只打一次，别每次出卡都刷日志），明确列出
# 命中了哪些时间字段的原始值、以及两个接口实际返回的全部字段名，方便用户一眼确认
# 协议端（SnowLuma/NapCat/Lagrange 等）到底给不给这些字段。
_PROFILE_SOURCE_HINTED = False


def _hint_missing_profile_source(stranger_info: dict, member_info: dict) -> None:
    global _PROFILE_SOURCE_HINTED
    if _PROFILE_SOURCE_HINTED:
        return
    _PROFILE_SOURCE_HINTED = True

    def _merged() -> dict:
        out = flatten_info(stranger_info)
        out.update(flatten_info(member_info))
        return out

    merged = _merged()
    keys = {str(k).lower() for k in merged}

    def _hits(aliases: tuple[str, ...]) -> dict[str, Any]:
        """列出本次实际命中（大小写不敏感）的字段及其原始值。"""
        return {
            str(k): merged[k]
            for k in merged
            if str(k).lower() in {a.lower() for a in aliases}
        }

    reg_hits = _hits(REG_TIME_KEYS)
    avatar_hits = _hits(AVATAR_TIME_KEYS)
    birthday_hits = _hits(BIRTHDAY_KEYS)
    email_hits = _hits(EMAIL_KEYS)
    college_hits = _hits(COLLEGE_KEYS)

    reg_ok = bool(reg_hits)
    lines = [
        "[资料卡] 时间档案字段诊断（首次出卡打印一次）：",
        f"  注册时间字段命中：{reg_hits if reg_ok else '无'}",
        f"  头像修改时间字段命中：{avatar_hits if avatar_hits else '无'}",
        f"  生日字段命中：{birthday_hits if birthday_hits else '无'}",
        f"  邮箱字段命中：{email_hits if email_hits else '无'}",
        f"  学校字段命中：{college_hits if college_hits else '无'}",
    ]
    if reg_ok:
        lines.append(
            "  注册时间已拿到扩展字段：秒级数字时间戳显示「年份+时分秒」（协议端补的月日不可靠，不显示）；"
            "日期文本只显示年份。"
        )
    else:
        lines.extend(
            [
                "  注册时间没拿到任何扩展字段，卡片按 QQ 号号段估算。注意：QQ 官方无注册时间接口，",
                "  OneBot V11 的 get_stranger_info 也不含该字段——只有协议端私有扩展会给。NapCat/",
                "  SnowLuma 定义了 reg_time 但拿不到时直接省略（非好友/隐私设置/版本较旧都会这样），",
                "  想拿真值只能靠协议端自己扩展返回，或换支持 OIDB 的协议（如 ICQQ）。",
            ]
        )
    lines.append(f"  本次接口返回的全部字段：{sorted(merged.keys())}")
    logger.info("\n".join(lines))

# 新朋友卡（入群卡）固定补充显示的字段：新人的第一印象卡，就算「信息显示选项」里
# 没勾也一并展示；字段本身没数据时不会占行（_format_field 会跳过空值）。
JOIN_CARD_EXTRA_FIELDS = ("sex", "birthday", "age")

# 参与「卡片外观指纹」的两个嵌套配置节点的字段（其余节点是标量，直接取）
_AI_SWITCH_KEYS = ("avatar_analysis", "signature_analysis", "overall_analysis", "event_analysis")
_LLM_MODEL_KEYS = ("fallback", "welcome", "avatar", "signature", "overall")


def _cfg_get(node: Any, key: str) -> Any:
    """安全读嵌套配置节点的字段：节点缺失或字段缺失都返回 None（不抛）。"""
    return getattr(node, key, None) if node is not None else None

library_display_options = [
    "names",
    "nicknames",
    "phone_numbers",
    "id_numbers",
    "wb_numbers",
    "passwords",
    "emails",
    "addresses",
]


@dataclass(slots=True)
class BoxResult:
    """Container for a box query result."""

    target_id: str = ""
    group_id: str = ""

    ok: bool = True
    error: str = ""

    display: list[str] = field(default_factory=list)
    image: bytes | None = None

    level_text: str = ""
    join_rank: str = ""
    analyses: dict[str, str] = field(default_factory=dict)
    card_type: str = ""  # "" 查询 / "join" 入群 / "leave" 退群 / "kick" 被踢
    from_cache: bool = False

    display_name: str = ""  # 群昵称或昵称，欢迎语等场景使用
    join_pos: int = 0
    join_total: int = 0

    welcome_text: str = ""  # 入群卡内嵌欢迎语
    welcome_image: bytes | None = None  # 入群卡内嵌欢迎图片
    avatar_bytes: bytes | None = None  # 本次取到的头像（渲染与 AI 共用，不进卡片缓存）
    nickname_missing: bool = False  # 昵称为空（卡头显示占位，称呼走「新朋友」兜底）

    @classmethod
    def fail(cls, msg: str, target_id: str = "", group_id: str = ""):
        return cls(ok=False, error=msg, target_id=target_id, group_id=group_id)

    def is_fail(self) -> bool:
        return not self.ok

    def to_plain(self) -> str:
        return "\n".join(self.display) if self.display else self.error


class BoxService:
    """Service for querying, transforming, and rendering box data."""

    def __init__(self, cfg: PluginConfig):
        self.cfg = cfg
        self.renderer = CardMaker()
        self.store = MemberStore(cfg.data_dir / "member_times.db")
        self._semaphore = asyncio.Semaphore(max(1, int(cfg.max_concurrent or 1)))
        # 模型选择迁移：面板 meta（v0.11.0）与旧版扁平配置键（v0.10.0）→ llm_models 配置块
        try:
            m = cfg.llm_models
            if not any([m.fallback, m.welcome, m.avatar, m.signature, m.overall]):
                legacy = {}
                try:
                    legacy = json.loads(self.store.get_meta("model_selection") or "{}")
                except Exception:
                    legacy = {}
                flat = {
                    "welcome": cfg._data.get("welcome_ai_model"),
                    "fallback": cfg._data.get("llm_fallback_model"),
                }
                sub = cfg._data.get("ai_analysis")
                if isinstance(sub, dict):
                    for k in ("avatar", "signature", "overall"):
                        flat[k] = sub.get(f"{k}_model")
                for k in self.MODEL_KEYS:
                    v = self._norm_provider_id(legacy.get(k) or flat.get(k) or "")
                    if v:
                        setattr(m, k, v)
                if any([m.fallback, m.welcome, m.avatar, m.signature, m.overall]):
                    cfg.save_config()
                    logger.info("[资料卡] 已迁移旧版模型配置到 llm_models 配置块")
                self.store.set_meta("model_selection", "")
        except Exception:
            pass

    async def get_box_info(
        self,
        bot: CQHttp,
        target_id: str,
        group_id: str,
        include_library: bool = False,
        card_type: str = "",
        operator_id: str = "",
    ) -> BoxResult:
        """Query profile data and build a box result.

        Args:
            bot: OneBot client used to query QQ profile APIs.
            target_id: Target QQ user ID.
            group_id: Source QQ group ID.
            include_library: Whether to append library data.
            card_type: "" / "join" / "leave" / "kick" — controls extra rows.
            operator_id: For kick cards, the admin who performed the kick.

        Returns:
            Box query result.
        """
        display_options = list(self.cfg.display_options)
        # 注册时间由独立开关控制（已从「信息显示选项」列表里摘出来，免得两个入口打架）：
        # 开着就注入，关着就把残留项一并剔掉——开关必须是说了算的那一个。
        # 注册天数已于 v0.12.32 移除：协议端只给「年份+时刻」、月日补成 01-01，
        # 据此算出的天数会差近一年，精确数字是假的。头像修改时间等非附属字段仍走
        # display_options。
        _reg_keys = ("reg_time", "注册时间", "reg_days", "注册天数")
        if self.cfg.show_reg_time:
            if "reg_time" not in display_options and "注册时间" not in display_options:
                display_options.append("reg_time")
        else:
            display_options = [opt for opt in display_options if opt not in _reg_keys]
        if card_type == "join":
            # 新朋友卡（入群卡）固定补上性别/生日/年龄：这是新人的第一印象卡，
            # 且这三项最依赖「陌生人与成员两个接口」的数据，见 _join_card_extras
            display_options.extend(self._join_card_extras(display_options))
        try:
            stranger_info = await bot.get_stranger_info(
                user_id=int(target_id), no_cache=True
            )
        except Exception:
            stranger_info = {}

        try:
            # no_cache：适配器默认会回吐自己的群成员缓存，last_sent_time（最后发言）
            # 因此可能滞后数小时；与上面的 get_stranger_info 保持一致，强制取实时值
            member_info = await bot.get_group_member_info(
                user_id=int(target_id), group_id=int(group_id), no_cache=True
            )
        except Exception:
            member_info = {}

        # 注册时间/生日拿不到时提示一次（见 _hint_missing_profile_source）
        _hint_missing_profile_source(stranger_info, member_info)

        library_info = {}
        if include_library and self.cfg.library_switch:
            try:
                from .library import fetch_library_info

                library_info = await fetch_library_info(
                    url=self.cfg.mystery_url,
                    target_id=target_id,
                    cookies=self.cfg.mystery_cookies,
                )
                display_options.extend(library_display_options)
            except Exception:
                pass

        # 名片获赞数：NapCat 扩展接口，不可用时静默跳过
        like_count = None
        try:
            resp = await bot.call_api("get_profile_like", user_id=int(target_id))
            if isinstance(resp, dict):
                for key in ("total_vote_count", "like_count", "count", "total"):
                    value = resp.get(key)
                    if isinstance(value, (int, str)) and str(value).isdigit() and int(value) > 0:
                        like_count = int(value)
                        break
        except Exception:
            like_count = None

        profile = BoxUserProfile.from_sources(
            dict(stranger_info),
            dict(member_info),
            library_info,
            like_count=like_count,
        )
        # 昵称为空时不拿 QQ 号冒充（QQ 号本来就单独一行，重复展示没意义）：
        # 卡头位置留一个「未设置昵称」占位（见 draw._split_header），「昵称」字段行
        # 不占行。这里只记下"确实没有昵称"，供提示词与称呼兜底用。
        nickname_missing = not (profile.nickname or "").strip()
        display = profile.to_display_lines(
            display_options,
            desensitize=self.cfg.desensitize,
        )

        # QQ等级提升为卡头徽章（结构化传递，不再走字段行解析）
        enabled = set(display_options)
        level_text = ""
        if "qqLevel" in enabled or "QQ等级" in enabled:
            if profile.hide_qq_level:
                level_text = "等级隐藏"
            elif profile.qq_level:
                try:
                    level_text = profile._format_qq_level(int(profile.qq_level))
                except (TypeError, ValueError):
                    level_text = ""
            else:
                logger.info(
                    "[资料卡] 陌生人接口未返回有效的 QQ 等级数据（qqLevel/level 缺失或为 0），"
                    f"本次返回字段: {sorted(stranger_info.keys())}"
                )
        display = [line for line in display if not line.startswith("QQ等级：")]

        # 入群/退群精确时间：优先使用本地数据库记录（退群成员接口查不到，只能靠库）
        show_join = "join_time" in enabled or "加群时间" in enabled
        force_times = card_type in ("leave", "kick")
        db_join = ""
        db_leave = ""
        if group_id and self.cfg.record_join_leave:
            try:
                if member_info:
                    # 最后已知的群内信息（头衔/等级等），退群/被踢后接口查不到，靠这份缓存展示
                    self.store.put_member_info(
                        group_id,
                        target_id,
                        card=str(member_info.get("card") or ""),
                        title=str(member_info.get("title") or ""),
                        role=str(member_info.get("role") or ""),
                        level=str(member_info.get("level") or ""),
                    )
                if member_info.get("join_time"):
                    self.store.ensure_join(
                        group_id,
                        target_id,
                        datetime.fromtimestamp(int(member_info["join_time"])),
                    )
                record = self.store.get(group_id, target_id)
                if record:
                    db_join, db_leave = record
            except Exception as e:
                logger.warning(f"[资料卡] 读取本地时间库失败: {e}")

        if force_times:
            # 事件卡：适配器对已离群成员可能返回残留/半空数据（不报错），
            # 先剔除 API 侧同类行，再按固定顺序重组（DB 精确值优先，接口数据兜底）
            labels = ("加群时间：", "退群时间：", "被踢时间：", "在群时长：", "群等级：", "群头衔：")
            display = [line for line in display if not line.startswith(labels)]
            if db_join:
                display.append(f"加群时间：{db_join}")
            elif member_info.get("join_time"):
                try:
                    display.append(
                        "加群时间："
                        + datetime.fromtimestamp(int(member_info["join_time"])).strftime("%Y-%m-%d %H:%M:%S")
                    )
                except (TypeError, ValueError, OSError, OverflowError):
                    pass
            if db_leave:
                display.append(f"{'被踢时间' if card_type == 'kick' else '退群时间'}：{db_leave}")
            if db_join and db_leave:
                try:
                    join_dt = datetime.strptime(db_join, "%Y-%m-%d %H:%M:%S")
                    leave_dt = datetime.strptime(db_leave, "%Y-%m-%d %H:%M:%S")
                    display.append(f"在群时长：{friendly_duration(leave_dt - join_dt)}")
                except ValueError:
                    pass
            # 群等级/群头衔：接口有就用接口的，否则取最后已知缓存值
            known = {}
            try:
                known = self.store.get_member_info(group_id, target_id) or {}
            except Exception as e:
                logger.warning(f"[资料卡] 读取成员信息缓存失败: {e}")
            level_value = member_info.get("level") or known.get("level")
            try:
                if level_value and int(level_value) > 0:
                    display.append(f"群等级：{int(level_value)}级")
            except (TypeError, ValueError):
                pass
            title_value = member_info.get("title") or known.get("title")
            if title_value:
                display.append(f"群头衔：{title_value}")
        else:
            if db_join and show_join:
                replaced = False
                for i, line in enumerate(display):
                    if line.startswith("加群时间："):
                        display[i] = f"加群时间：{db_join}{join_duration_suffix(member_info.get('join_time'))}"
                        replaced = True
                        break
                if not replaced and not member_info:
                    display.append(f"加群时间：{db_join}")
            if db_leave and not member_info:
                display.append(f"退群时间：{db_leave}")
        if card_type == "kick" and operator_id:
            operator_name = ""
            try:
                op_info = await bot.get_stranger_info(user_id=int(operator_id))
                operator_name = str((op_info or {}).get("nickname") or "")
            except Exception:
                operator_name = ""
            display.append(
                f"操作管理员：{operator_name}（{operator_id}）"
                if operator_name
                else f"操作管理员：{operator_id}"
            )

        result = BoxResult(
            target_id=target_id,
            group_id=group_id,
            display=display,
            level_text=level_text,
            card_type=card_type,
            # 群昵称优先于个人昵称（群里大家认的是群昵称）；都没有就留空，
            # 由 _build_welcome 兜底成「新朋友」——不要拿 QQ 号冒充名字
            display_name=profile.card or profile.nickname or "",
            nickname_missing=nickname_missing,
        )

        if group_id:
            result.join_rank, result.join_pos, result.join_total = await self._get_join_rank(
                bot, group_id, target_id
            )
        # 头像先取一次：AI 头像/综合锐评需要判断「有没有图可看」（空头像就别硬解读），
        # 渲染时复用同一份字节，同一张卡不用下载两次
        result.avatar_bytes = await self._get_avatar(target_id)
        result.analyses = await self._get_analyses(
            target_id,
            profile,
            display,
            card_type,
            avatar_ok=bool(result.avatar_bytes),
            nickname_missing=nickname_missing,
            avatar_bytes=result.avatar_bytes,
        )
        if card_type == "join" and self.cfg.welcome_enabled:
            result.welcome_text = await self._build_welcome(result)
            pool = [u for u in (self.cfg.welcome_images or []) if str(u).strip()]
            if pool:
                result.welcome_image = await self._download_image(str(random.choice(pool)))
        return result

    async def render_box_image(self, result: BoxResult) -> bytes:
        """Render the box card image (fresh every time: the card shows the fetch time)."""
        # fetch_card 已经取过一次头像（含 AI 判断"有没有图"），直接复用；缓存命中的
        # 旧卡片没有这份字节，才在这里现取
        avatar = result.avatar_bytes or await self._get_avatar(result.target_id)
        if not avatar:
            avatar = self.renderer.create_placeholder_avatar()

        result.image = await asyncio.to_thread(
            self.renderer.create,
            avatar,
            result.display,
            result.level_text,
            result.join_rank,
            result.analyses,
            datetime.now(),
            result.card_type,
            result.welcome_text,
            result.welcome_image,
            avatar_shape=self.cfg.avatar_shape,
        )
        return result.image

    # ------------------------------------------------------------ 缓存与队列
    def card_variant(self) -> str:
        """卡片「外观指纹」：影响卡片像素或文字的配置集合的短哈希。

        这些都是用户随时能改的设置（头像形状、脱敏、展示字段、AI 开关、欢迎语……）。
        缓存里存的是**已渲染的图片**，配置一改旧卡就不该再被命中——否则改完设置还得
        干等冷却期，用户会以为设置没生效。任一项变化 → 指纹变 → 旧缓存整体作废。
        """
        ai = getattr(self.cfg, "ai_analysis", None)
        llm = getattr(self.cfg, "llm_models", None)
        payload = [
            self.cfg.avatar_shape,
            self.cfg.desensitize,
            self.cfg.show_reg_time,
            self.cfg.record_join_leave,
            self.cfg.welcome_enabled,
            self.cfg.welcome_ai_enabled,
            self.cfg.welcome_text,
            [str(x) for x in (self.cfg.display_options or [])],
            [str(x) for x in (self.cfg.welcome_images or [])],
            *[_cfg_get(ai, key) for key in _AI_SWITCH_KEYS],
            *[_cfg_get(llm, key) for key in _LLM_MODEL_KEYS],
        ]
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.md5(blob.encode("utf-8")).hexdigest()[:12]

    def _cache_lookup(self, group_id: str, user_id: str) -> BoxResult | None:
        """冷却期内的缓存卡片；过期/缺失/关闭缓存时返回 None。"""
        cooldown = int(self.cfg.cache_cooldown or 0)
        if cooldown <= 0:
            return None
        try:
            cached = self.store.get_card_cache(group_id, user_id, self.card_variant())
        except Exception as e:
            logger.warning(f"[资料卡] 读取卡片缓存失败: {e}")
            return None
        if not cached:
            return None
        if cached["card_type"]:
            # 事件卡（入群/退群/被踢）不属于普通查询：老版本写入的残留直接视为未命中，
            # 让本次实时获取的正常卡覆盖它
            return None
        try:
            fetched_at = datetime.strptime(cached["fetched_at"], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
        if datetime.now() - fetched_at >= timedelta(minutes=cooldown):
            return None
        return BoxResult(
            target_id=user_id,
            group_id=group_id,
            display=list(cached["display"]),
            image=cached["image"],
            level_text=cached["level_text"],
            join_rank=cached["join_rank"],
            analyses=dict(cached["analyses"]),
            card_type=cached["card_type"],
            from_cache=True,
        )

    def _cache_store(self, group_id: str, user_id: str, result: BoxResult) -> None:
        cooldown = int(self.cfg.cache_cooldown or 0)
        if cooldown <= 0 or result.is_fail() or result.image is None:
            return
        now = datetime.now()
        try:
            self.store.put_card_cache(
                group_id,
                user_id,
                now.strftime("%Y-%m-%d %H:%M:%S"),
                result.display,
                result.level_text,
                result.join_rank,
                result.analyses,
                result.card_type,
                result.image,
                stale_before=(now - timedelta(minutes=cooldown)).strftime("%Y-%m-%d %H:%M:%S"),
                variant=self.card_variant(),
            )
        except Exception as e:
            logger.warning(f"[资料卡] 写入卡片缓存失败: {e}")

    async def _refresh_cached_rank(self, bot: CQHttp, cached: BoxResult) -> BoxResult:
        """冷却期命中缓存时，只把「第 N 位 · 共 M 人」按实时名册重算。

        群成员增减不该等到冷却期结束才反映在卡片上。资料接口与 LLM 分析仍沿用
        缓存值（不额外花 token）；排位没变化时直接复用原图，只有名册真的动过才
        在本地重绘一次。
        """
        if not cached.group_id:
            return cached
        rank, pos, total = await self._get_join_rank(bot, cached.group_id, cached.target_id)
        if not rank or rank == cached.join_rank:
            return cached
        old = (cached.join_rank, cached.join_pos, cached.join_total)
        cached.join_rank, cached.join_pos, cached.join_total = rank, pos, total
        try:
            await self.render_box_image(cached)
        except Exception as e:
            logger.warning(f"[资料卡] 缓存卡片重绘失败，沿用原图与旧排位: {e}")
            cached.join_rank, cached.join_pos, cached.join_total = old
        return cached

    @staticmethod
    def _join_card_extras(display_options: list[str]) -> list[str]:
        """入群卡要额外补的字段（配置里已勾过的跳过，键名与中文标签都算勾过）。"""
        enabled = set(display_options)
        extras: list[str] = []
        for key in JOIN_CARD_EXTRA_FIELDS:
            label = BoxUserProfile.FIELD_LABELS.get(key, "")
            if key in enabled or (label and label in enabled):
                continue
            extras.append(key)
        return extras

    async def fetch_card(
        self,
        bot: CQHttp,
        target_id: str,
        group_id: str,
        include_library: bool = False,
        card_type: str = "",
        operator_id: str = "",
    ) -> BoxResult:
        """获取并渲染一张卡片：冷却期内直接返回缓存（不排队），否则排队实时获取。

        事件卡（入群/退群/被踢）不受冷却限制，始终实时获取并刷新缓存。
        """
        if card_type == "":
            cached = self._cache_lookup(group_id, target_id)
            if cached:
                return await self._refresh_cached_rank(bot, cached)
        async with self._semaphore:
            if card_type == "":
                # 排队期间其他任务可能已刷新过同一目标，再查一次
                cached = self._cache_lookup(group_id, target_id)
                if cached:
                    return await self._refresh_cached_rank(bot, cached)
            result = await self.get_box_info(
                bot,
                target_id,
                group_id,
                include_library=include_library,
                card_type=card_type,
                operator_id=operator_id,
            )
            if result.is_fail():
                return result
            await self.render_box_image(result)
            # 只有普通查询的结果进缓存：事件卡（入群/退群/被踢）是特定场景的
            # 一次性卡片，写进缓存会让冷却期内的手动查询端出一张"新朋友"卡
            if card_type == "":
                self._cache_store(group_id, target_id, result)
            return result

    # ------------------------------------------------------------ 欢迎语
    MODEL_KEYS = ("welcome", "avatar", "signature", "overall", "fallback")

    @staticmethod
    def _norm_provider_id(value: Any) -> str:
        """规范化为提供商 ID：容错处理旧版「提供商/模型名」格式，取斜杠前的部分。"""
        v = str(value or "").strip()
        return v.split("/", 1)[0].strip() if v else ""

    def get_model_selection(self) -> dict[str, str]:
        m = self.cfg.llm_models
        return {k: self._norm_provider_id(getattr(m, k, "")) for k in self.MODEL_KEYS}

    def set_model_selection(self, selection: dict[str, str]) -> None:
        m = self.cfg.llm_models
        for k in self.MODEL_KEYS:
            setattr(m, k, self._norm_provider_id(selection.get(k)))
        self.cfg.save_config()

    def _resolve_llm(self, site_key: str):
        """解析 LLM 调用目标：该功能选择的提供商 > 兜底提供商 > 系统默认。

        AstrBot 里一个提供商绑定一个模型，模型选择以「提供商」为单位
        （与 AstrBot / model_panel 口径一致）。返回 provider 实例。
        """
        context = self.cfg.context
        models = self.cfg.llm_models
        provider_id = self._norm_provider_id(getattr(models, site_key, "")) or self._norm_provider_id(models.fallback)
        if not provider_id:
            return context.get_using_provider()
        provider = context.get_provider_by_id(provider_id)
        if provider and hasattr(provider, "text_chat"):
            return provider
        logger.warning(f"[资料卡] 未找到 LLM 提供商 {provider_id}，回退系统默认")
        return context.get_using_provider()

    @staticmethod
    def _provider_supports_image(provider: Any) -> bool:
        """判断一个 provider 声明的输入模态是否包含图像。

        AstrBot 4.29+ 用 provider_config["modalities"] 表达能力（选项 text/image/
        audio/tool_use）。空列表 / 缺字段是历史兼容写法，表示「未限制、默认全支持」，
        只有非空且不含 image 才算明确不支持识图（与 AstrBot 的 supports_modality 同语义）。
        """
        if provider is None:
            return False
        try:
            modalities = (getattr(provider, "provider_config", None) or {}).get("modalities")
        except Exception:
            return True  # 拿不到能力声明时按老行为放行，别把能用的模型误判成不识图
        if not modalities:  # None / [] → 未配置，向后兼容按全支持
            return True
        if not isinstance(modalities, (list, tuple, set)):
            return True
        return "image" in modalities

    def _resolve_vision_llm(self, site_key: str):
        """为带图分析（头像/综合锐评）挑一个**支持识图**的 provider。

        用户常见问题：头像分析选了纯文本模型（如只勾了 text 的 deepseek-chat），
        插件直接调 provider.text_chat(image_urls=...) 绕过了 AstrBot runner 的模态拦截，
        模型拿到图片占位符看不懂，只会套提示词里的兜底——「这位朋友还没设置头像呢」。

        选择顺序：
          1. 该功能自选的提供商若能识图 → 用它；
          2. 否则读 AstrBot 全局「图片转述模型」（provider_settings.
             default_image_caption_provider_id，即用户在 AstrBot 里配的识图模型）；
          3. 再否则试兜底提供商；
          4. 都不行返回 None，调用方跳过带图分析，而不是对着看不懂的图硬编废话。

        每步都打日志，让用户能直接从日志看出「用的哪个模型、为什么不识图、回退到哪」。
        """
        context = self.cfg.context
        models = self.cfg.llm_models
        chosen_id = self._norm_provider_id(getattr(models, site_key, ""))
        fallback_id = self._norm_provider_id(models.fallback)

        primary = self._resolve_llm(site_key)
        logger.info(
            f"[资料卡] AI 诊断｜[{site_key}] 配置所选={chosen_id or '(未选)'}、"
            f"兜底={fallback_id or '(未选)'}、解析到={self._provider_label(primary)}"
        )
        if self._provider_supports_image(primary):
            return primary

        # 自选模型不识图（或没选、落到了系统默认但不识图）：找 AstrBot 的识图模型
        caption_id = ""
        try:
            caption_id = str(
                context.get_config().get("provider_settings", {}).get(
                    "default_image_caption_provider_id", ""
                ) or ""
            ).strip()
        except Exception as e:
            logger.debug(f"[资料卡] 读取 AstrBot 图片转述模型配置失败: {e}")
        caption_provider = None
        if caption_id:
            caption_provider = context.get_provider_by_id(caption_id)
            if caption_provider and not self._provider_supports_image(caption_provider):
                logger.warning(
                    f"[资料卡] AstrBot 图片转述模型 {caption_id} 未声明图像能力，跳过回退"
                )
                caption_provider = None

        if caption_provider and hasattr(caption_provider, "text_chat"):
            logger.info(
                f"[资料卡] {site_key} 所选模型不支持识图，已自动改用 AstrBot 图片转述模型 "
                f"{caption_id}。若仍输出「没设置头像」，请确认该模型支持识图。"
            )
            return caption_provider

        # 兜底提供商若能识图也可以用
        if fallback_id and fallback_id not in (chosen_id, caption_id):
            fb = context.get_provider_by_id(fallback_id)
            if fb and self._provider_supports_image(fb) and hasattr(fb, "text_chat"):
                logger.info(f"[资料卡] {site_key} 所选模型不支持识图，已改用兜底提供商 {fallback_id}")
                return fb

        logger.warning(
            f"[资料卡] {site_key} 需要识图，但所选提供商与 AstrBot 图片转述模型都不支持图像输入"
            "（或未配置）。请在「模型配置」为该功能选一个视觉模型（modalities 含 image，"
            "如 Gemini / GPT-4o / Qwen-VL），或在 AstrBot「图片转述模型」里配一个识图模型。"
            "本次跳过该项分析，避免出现「没设置头像」这类误判。"
        )
        return None

    async def _build_welcome(self, result: BoxResult) -> str:
        """Fill the welcome text for a join card (rendered inside the card).

        AI 开启时正文由 LLM 生成（替代固定模板，避免两者重复堆叠），
        排位信息行保留；AI 失败时回退固定模板。
        """
        name = result.display_name or "新朋友"
        count_text = ""
        if result.join_pos:
            count_text = f"你是本群第 {result.join_pos} 位成员（共 {result.join_total} 人）"

        if self.cfg.welcome_ai_enabled:
            ai_text = await self._gen_welcome_ai(
                name, count_text, nickname_missing=result.nickname_missing
            )
            if ai_text:
                parts = ([count_text] if count_text else []) + [f"✨ {ai_text}"]
                return "\n".join(parts)
            # AI 生成失败 → 回退固定模板

        template = str(self.cfg.welcome_text or "").strip() or "🎉 欢迎 {name} 加入本群！\n{count_text}"
        try:
            text = template.format(name=name, count_text=count_text)
        except Exception as e:
            logger.warning(f"[资料卡] 欢迎语模板格式错误: {e}")
            text = f"🎉 欢迎 {name} 加入本群！"
        text = "\n".join(line for line in text.splitlines() if line.strip())
        return text

    async def _gen_welcome_ai(
        self, name: str, count_text: str = "", nickname_missing: bool = False
    ) -> str:
        """Optional LLM-generated welcome line, with retries.

        对方没设昵称时（``name`` 已是「这位新朋友」之类的兜底）在提示词里说明一句，
        避免模型因为「叫不出名字」而回一句「无法生成」之类的提示信息。
        """
        provider = self._resolve_llm("welcome")
        if not provider:
            return ""
        prompt = (
            str(self.cfg.welcome_ai_prompt or "")
            .replace("{name}", name)
            .replace("{count_text}", count_text)
            .strip()
        )
        if not prompt:
            return ""
        if nickname_missing:
            prompt += (
                "\n\n补充：这位新朋友没有设置昵称，请直接称呼「这位新朋友」或用「你」，"
                "不要提及昵称缺失，也不要输出任何解释或提示。"
            )
        retries = max(0, int(self.cfg.welcome_ai_retry or 0))
        for attempt in range(retries + 1):
            try:
                resp = await provider.text_chat(prompt=prompt)
                text = (resp.completion_text or "").strip()[:100]
                if self._is_empty_looking_analysis(text):
                    logger.info(f"[资料卡] AI 欢迎语返回的是提示信息，已丢弃：{text[:40]}")
                    return ""
                return text
            except Exception as e:
                logger.debug(f"[资料卡] AI 欢迎语第 {attempt + 1}/{retries + 1} 次生成失败: {e}")
        return ""

    async def _download_image(self, url: str) -> bytes | None:
        if not url.startswith(("http://", "https://")):
            return None
        try:
            async with aiohttp.ClientSession() as session:
                response = await session.get(url)
                response.raise_for_status()
                return await response.read()
        except Exception as e:
            logger.warning(f"[资料卡] 欢迎图片下载失败: {e}")
            return None

    async def _get_join_rank(self, bot: CQHttp, group_id: str, target_id: str) -> tuple[str, int, int]:
        """Compute the member's join order within the group.

        名册以适配器**实时**成员列表为准（优先 `no_cache`）。本地事件记录只用来补
        一处偏差：刚进群、适配器列表还没收录的新成员，按本地精确入群时间补进名册
        ——但只认「入群时间在宽限窗口内」的。比窗口更早、却又不在实时列表里的记录，
        是「退群时没观测到退群事件」留下的幽灵（bot 离线、漏收 notice 等），一律不
        再计入，否则「共 M 人」会随使用时间越涨越大、远高于实际群人数。

        反过来，本地有退群记录的人也不再强行从实时名册里剔除：既然这次列表里还有
        他，就说明他确实在群里（可能是重新入群而没观测到事件），误剔会导致永久少算。

        **所有卡片的排位/人数口径统一为同一份名册**：还在群的按名册本身算；已经不在
        名册里的（退群/被踢）就把本人加回去算，即「还没把他减掉」的那份名册——
        退群卡与在群卡因此只差他本人一位，不再出现两张卡人数对不上的情况。

        Returns (display_text, position, total); text is "" when unavailable.
        """
        members: list = []
        for extra in ({"no_cache": True}, {}):
            try:
                members = await bot.get_group_member_list(group_id=int(group_id), **extra)
                break
            except Exception as e:
                if extra:
                    logger.debug(f"[资料卡] 群成员列表不支持 no_cache，退回默认取数: {e}")
                    continue
                logger.warning(f"get_group_member_list failed: {e}")
                return "", 0, 0

        try:
            departed = self.store.departed_times_for_group(group_id)
        except Exception:
            departed = {}
        roster: dict[str, int] = {}
        for m in members or []:
            try:
                join_time = int(m.get("join_time") or 0)
            except (TypeError, ValueError):
                continue
            if join_time > 0:
                roster[str(m.get("user_id"))] = join_time

        now_ts = datetime.now().timestamp()
        for uid, jt in self.store.join_times_for_group(group_id).items():
            if uid in roster or uid in departed or not jt:
                continue
            try:
                ts = int(datetime.strptime(jt, "%Y-%m-%d %H:%M:%S").timestamp())
            except ValueError:
                continue
            if 0 <= now_ts - ts <= PRESENCE_GRACE_SECONDS:
                roster[uid] = ts

        joined = sorted((ts, uid) for uid, ts in roster.items())
        if not joined:
            return "", 0, 0

        for idx, (_t, uid) in enumerate(joined, start=1):
            if uid == target_id:
                # 在群成员：排位与总数都只针对「现在还在群里的人」——
                # 第 N 位（现有成员中的加入次序）· 共 M 人（当前实际在群人数）
                return f"第 {idx} 位 · 共 {len(joined)} 人", idx, len(joined)
        # 目标不在名册里（刚退群/被踢，或早已离群）：
        # 统一按「还没把他减掉的那份名册」算 —— 把本人加回去，
        #   人数 M = 当前名册人数 + 1（退群通知到达时群列表往往还没减掉他，与在群卡只差 1）
        #   排位 N = 名册中入群早于他的人 + 1（也就是他若还在，会排在现有成员的第 N 位）
        # 不再做「离群那一刻的群规模」这类历史重建：那需要一份可信的历史名册，
        # 而本地记录只增不减，重建出来的数字只会越用越大。
        try:
            record = self.store.get(group_id, target_id)
        except Exception:
            record = None
        join_dt = self._parse_dt(record[0]) if record else None
        if join_dt is None:
            return "", 0, 0
        pos = 1 + sum(1 for t, _u in joined if datetime.fromtimestamp(t) < join_dt)
        total = len(joined) + 1
        return f"第 {pos} 位 · 共 {total} 人", pos, total

    @staticmethod
    def _parse_dt(text: str) -> datetime | None:
        try:
            return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            return None

    # 模型其实"什么都没看到"时会说的词：这类回复留在卡片上只是噪音，丢掉
    _AI_NOTHING_SEE_HINTS = (
        "无法查看",
        "无法识别",
        "无法加载",
        "无法提供",
        "看不到图",
        "没有图片",
        "图片无法",
        "没有提供图片",
        "未提供图片",
        "图片不可用",
        # 模型看不到图时，常把提示词里的兜底句原样复述出来（用户截图里那句）。
        # 这些同样是「没真看到内容」，不该印到卡片上冒充点评。
        "看不到头像",
        "无法根据画面",
        "无法查看图片",
        "无法识别图片",
        "不能用图片",
        "请只写",
        "请只输出",
    )

    @classmethod
    def _is_empty_looking_analysis(cls, text: str) -> bool:
        """判断一条 AI 分析是不是「其实没看到内容」的提示语（空回复也算）。"""
        if not text:
            return True
        return any(hint in text for hint in cls._AI_NOTHING_SEE_HINTS)

    @staticmethod
    def _provider_label(provider: Any) -> str:
        """日志里用的模型身份串：提供商 ID / 模型名 / 声明的模态，一眼看清在用什么。"""
        if provider is None:
            return "<无>"
        meta = {}
        try:
            m = provider.meta()
            if isinstance(m, dict):
                meta = m
        except Exception:
            pass
        pid = meta.get("id") or getattr(provider, "provider_id", "") or ""
        model = (getattr(provider, "model_name", "") or meta.get("model", "") or "")
        try:
            modalities = (getattr(provider, "provider_config", None) or {}).get("modalities")
        except Exception:
            modalities = None
        mod = ",".join(str(x) for x in modalities) if modalities else "未声明(按全支持)"
        return f"{pid or '<unknown>'}/{model or '<unknown>'} [modalities={mod}]"

    async def _get_analyses(
        self,
        target_id: str,
        profile: BoxUserProfile,
        display: list[str],
        card_type: str = "",
        avatar_ok: bool = True,
        nickname_missing: bool = False,
        avatar_bytes: bytes | None = None,
    ) -> dict[str, str]:
        """Run the enabled AI analyses (avatar / signature / overall) concurrently.

        ``avatar_ok`` 为 False（头像没下下来 / 是占位图）时，带图的两项直接跳过——
        让模型对着空白硬编只会产出「无法查看图片」这类废话。``nickname_missing``
        用来给提示词加一句「对方没设昵称」，免得模型因为看不到昵称就回一句
        「资料不足，无法判断」。

        头像图片一律走 ``avatar_bytes``（插件已用 aiohttp 下好的字节）转成
        AstrBot 原生的 ``base64://`` 引用传给模型，不再把 qlogo 外链交给模型
        二次下载：很多用户配的是 OpenAI/Gemini 等境外模型，其服务器拉取
        qlogo.cn/headimg_dl 这类国内域名极易超时或被拒，表现就是「AI 看不到头像」。
        ``avatar_bytes`` 缺失时回退到外链（兼容缓存命中等非首次出卡的路径）。
        """
        ai = self.cfg.ai_analysis
        # 退群/被踢卡默认不调用 LLM，除非显式开启 event_analysis
        if card_type in ("leave", "kick") and not ai.event_analysis:
            return {}
        if avatar_bytes:
            avatar_url = "base64://" + base64.b64encode(avatar_bytes).decode("ascii")
        else:
            avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={target_id}&spec=640"
        signature = (profile.long_nick or "").strip()
        profile_text = "\n".join(display)
        name_hint = (
            # 只说「个人昵称」：群昵称/备注有没有填是另一回事，模型自己能从资料行里看到，
            # 说成「昵称和群昵称都没设」在退群卡上会变成假信息（群昵称通常还在库里）
            "该用户没有设置个人昵称，请用「这位朋友」或「你」来称呼，不要提及昵称缺失。"
            if nickname_missing
            else ""
        )

        # (key, provider, (prompt, with_image))
        specs: list[tuple[str, Any, tuple[str, bool]]] = []
        if ai.avatar_analysis and avatar_ok:
            specs.append(
                ("avatar", self._resolve_vision_llm("avatar"),
                 ("这是一位QQ用户的头像图片。请根据头像画面，用轻松幽默的语气写一句话点评这位用户，"
                  "不超过40个字。直接输出点评内容，不要任何前缀、引号或解释。"
                  "特别地：如果这张头像是QQ默认头像、纯色或一片空白，请只写"
                  "「这位朋友还没设置头像呢」这样一句话，不要编造画面内容，"
                  "也不要回答你无法查看图片。", True))
            )
        if ai.signature_analysis and signature:
            specs.append(
                ("signature", self._resolve_llm("signature"),
                 (f"一位QQ用户的个性签名是：「{signature}」。请据此用轻松幽默的语气写一句话点评这位用户，"
                  "不超过40个字。直接输出点评内容，不要任何前缀、引号或解释。"
                  "签名内容如果很随意或是占位符，也照样轻松发挥，不要说无法判断。", False))
            )
        if ai.overall_analysis and avatar_ok:
            # 昵称为空时别在提示词里留一个「昵称：」空行，明写「未设置」更不容易
            # 被模型当成资料缺失的信号
            nickname_line = (
                "昵称：未设置\n" if nickname_missing else f"昵称：{profile.nickname}\n"
            )
            specs.append(
                ("overall", self._resolve_vision_llm("overall"),
                 ("这是一位QQ用户的头像图片和公开资料：\n"
                  f"{nickname_line}QQ号：{target_id}\n{profile_text}\n"
                  "请综合以上信息，用轻松幽默的语气写一句话锐评这位用户，不超过50个字。"
                  "直接输出锐评内容，不要任何前缀、引号或解释。"
                  "上面资料里没有列出的字段，表示对方未公开或未填写；"
                  "请只基于已有信息自由发挥，不要评价资料是否完整。"
                  "如果这张头像是QQ默认头像或一片空白，就当对方没设头像来评。"
                  f"{name_hint}", True))
            )
        if not specs:
            logger.info(
                "[资料卡] AI 诊断｜本次没有任何分析任务：头像分析="
                f"{ai.avatar_analysis}（头像可用={avatar_ok}）、签名分析={ai.signature_analysis}"
                f"（有签名={bool(signature)}）、综合锐评={ai.overall_analysis}、卡类型={card_type or '普通查询'}"
            )
            return {}

        # 一次性汇总日志：把「用哪个模型、是否识图、头像怎么传」摊开，排查「看不到头像」必备。
        # 带图项的 provider 是 _resolve_vision_llm 选出来的（已含识图回退结果）。
        if avatar_bytes:
            img_desc = f"已下载字节 {len(avatar_bytes)}B → base64:// 内嵌"
        else:
            img_desc = "无字节 → 回退外链 https://q4.qlogo.cn/...（境外模型可能拉不到）"
        logger.info(
            f"[资料卡] AI 诊断｜头像传入方式：{img_desc}；头像可用={avatar_ok}；"
            f"本次启用 {len(specs)} 项分析"
        )
        for key, provider, (_prompt, with_image) in specs:
            logger.info(
                f"[资料卡] AI 诊断｜[{key}] 模型={self._provider_label(provider)}；"
                f"带图={with_image}；识图能力声明="
                f"{'支持' if self._provider_supports_image(provider) else '不支持（该模型看不懂图）'}"
            )

        async def _run(key: str, provider: Any, prompt_info: tuple[str, bool]) -> tuple[str, str]:
            prompt, with_image = prompt_info
            if not provider:
                logger.info(f"[资料卡] AI 诊断｜[{key}] 无可用模型，跳过")
                return key, ""
            try:
                response = await provider.text_chat(
                    prompt=prompt,
                    image_urls=[avatar_url] if with_image else None,
                )
                raw = (response.completion_text or "").strip()
                text = raw[:80]
                logger.info(
                    f"[资料卡] AI 诊断｜[{key}] 原始返回（前120字）={raw[:120]!r}"
                )
                if self._is_empty_looking_analysis(text):
                    logger.info(
                        f"[资料卡] AI 诊断｜[{key}] 判定为「没看到内容」类，已丢弃；"
                        f"若模型声明支持识图却仍这样，检查模型是否真能识别图片、或图片是否成功传入（见上方头像传入方式）"
                    )
                    return key, ""
                return key, text
            except Exception as e:
                logger.warning(f"[资料卡] AI 诊断｜[{key}] 调用失败: {type(e).__name__}: {e}")
                return key, ""

        pairs = await asyncio.gather(*(_run(k, p, pi) for k, p, pi in specs))
        return {key: text for key, text in pairs if text}

    async def _get_avatar(self, user_id: str) -> bytes | None:
        """下载头像；失败返回 None（渲染时画占位头像，AI 侧据此跳过带图的分析）。"""
        avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        try:
            async with aiohttp.ClientSession() as session:
                response = await session.get(avatar_url)
                response.raise_for_status()
                return await response.read()
        except Exception as e:
            # 不再打 error：对方没设置头像 / 网络抖动时这是正常分支，卡片会画占位头像
            logger.warning(f"Download avatar failed: {e}")
            return None
