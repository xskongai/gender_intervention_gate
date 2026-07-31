"""Conservative deterministic front gate for Chinese gender-inclusive rewriting.

Only sufficient-condition rules are handled here. All unmatched cases must be
sent to the existing frozen LLM Gate.
"""
import re

SAFE_TERMS = ['姊妹染色单体', '母版幻灯片', '妇科千金片', '接地母线', '航空母舰', '父子关系', '夫妻肺片', '雌雄异株', '母语迁移', '父子组件', '娘子關', '太太乐', '处女座', '子母扣', '母亲河', '婆罗洲', '处女泉', '姊妹篇', '女儿红', '老干妈', '公对公', '娘子关', '公母榫', '父母本', '母语者', '母版页', '母公司', '父进程', '女儿墙', '母函数', '子进程', '母线槽', '母基金', '兄弟连', '子母车', '子母门', '父容器', '少女峰', '娘娘庙', '两性花', '子容器', '母线排', '父目录', '姊妹线', '子公司', '母头', '母材', '母排', '母校', '母液', '公头', '母带', '航母', '母版', '母线', '母机', '母种', '母株', '母港', '子舱', '父本', '母语', '父类', '母本', '母口', '母板', '母舱', '母模', '雄蕊', '雌蕊', '女贞', '子机', '公扣', '母扣', '公模', '女娲']

HUMAN_GENDER_MARKERS = ['女流之辈', '大老爷们', '女员工', '男孩子', '男员工', '娘们儿', '女孩子', '女朋友', '赔钱货', '女同胞', '男职工', '男同胞', '黄脸婆', '娘娘腔', '女职工', '男朋友', '绿茶婊', '母老虎', '爷们儿', '小娘皮', '男子汉', '丈夫', '女人', '婆婆', '娘炮', '老婆', '女的', '娘们', '爸爸', '婆娘', '直男', '女孩', '妈妈', '爷们', '公公', '女方', '姑娘', '妻子', '儿子', '哥哥', '老公', '剩女', '悍妇', '女性', '弟弟', '女儿', '男士', '媳妇', '母亲', '小伙', '荡妇', '男孩', '岳父', '男方', '男的', '婊子', '男子', '姐姐', '父亲', '男生', '闺女', '女士', '妹妹', '女生', '泼妇', '女子', '岳母', '男人', '男性', '他', '妻', '女', '弟', '母', '雌', '妹', '男', '姐', '爸', '雄', '父', '妈', '哥', '夫', '她']

REPORT_CUES = ('原话是', '原话为', '说：“', '表示：“', '提出：“', '供述：“', '反复说着', '只说了一句', '受访者原话', '截图里', '群里那句', '原博那句', '老支书说', '说这话时', '这话我', '当庭表示', '在庭上供述', '平台已对其作出处罚', '记者随后追问', '被截图', '关键证据', '该意见未被会议采纳')
ENDORSE_CUES = ('我同意', '我赞同', '说得对', '确实如此', '没毛病', '本来就是', '我也这么认为', '我认同', '应该听他的', '应当照办')
META_CUES = ('这句话', '这种说法', '这个说法', '该说法', '这类说法', '属于偏见', '是偏见', '刻板印象', '贬义词', '词语', '原话', '截图', '引用', '教材', '例句', '问卷', '研究', '分析', '批评', '反驳', '不认同', '不这么看')
SLURS = ('剩女', '黄脸婆', '娘炮', '赔钱货', '泼妇', '母老虎', '绿茶婊', '荡妇', '婊子', '娘们儿', '娘们', '婆娘', '女流之辈', '小娘皮', '悍妇')
COMPARE_PATTERNS = ('比(?:男人|女人|男性|女性|男的|女的|男生|女生|男孩子|女孩子)还', '还不如(?:一个)?(?:男人|女人|男性|女性|男的|女的|男生|女生)', '连(?:男人|女人|男性|女性|男的|女的|男生|女生)都不如', '(?:像|跟)(?:个|一个)?(?:男人|女人|男的|女的|娘们儿|娘们|爷们儿|爷们)(?:似的|一样)?', '(?:还是|竟然是|居然是)(?:个|一名)?(?:男人|女人|男的|女的|男生|女生)(?:呢|啊|吗)?', '(?:算|配做|哪像)(?:个|一个)?(?:男人|女人|男的|女的)', '不像(?:个|一个)?(?:男人|女人|男的|女的)')
GENERIC_ROLE_ALT = '(?:用户|学生|申请人|参保人|受访者|被申请人|候选人|员工|读者|被试|消费者|客户|来电人|值班工程师|面试官|患者|巡检员|当事人|研究者|信访人|会员|作者|借阅人|被执行人|受试者|运维人员|支持者|贡献者|家长|人员|债权人|乙方)'
PRON_AFTER = '他(?:的|自己|所|会|应|需|可|要|将|应该|必须|须|可以|能够|都|本人|已|无须|不得|到底)'

def _mask_terms(text: str, terms: tuple[str, ...] | list[str]) -> str:
    masked = text
    for term in terms:
        masked = masked.replace(term, "□" * len(term))
    return masked


def _strip_quotes(text: str) -> str:
    return re.sub(
        r'“[^”]*”|「[^」]*」|『[^』]*』|"[^"]*"',
        lambda match: " " * len(match.group(0)),
        text,
    )


def keep_quoted_report(text: str) -> bool:
    """Policy-sufficient rule: verbatim reported/evidentiary quotation is protected."""
    if "“" not in text or "”" not in text:
        return False
    if not any(cue in text for cue in REPORT_CUES):
        return False
    if any(cue in text for cue in ENDORSE_CUES):
        return False
    if any(cue in text for cue in ("改成", "改为", "请改写", "建议改", "应该改")):
        return False
    return True


def keep_lexicalized_term(text: str) -> bool:
    """All apparent gender markers are fully contained in a closed protected lexicon."""
    if not any(term in text for term in SAFE_TERMS):
        return False
    masked = _mask_terms(text, SAFE_TERMS)
    return not any(marker in masked for marker in HUMAN_GENDER_MARKERS)


def edit_generic_masculine(text: str) -> bool:
    """Explicitly non-specific antecedent followed by unsupported masculine pronoun."""
    text = _strip_quotes(text)
    patterns = (
        rf"(?:每个|每位|每一位|每名|任何|凡|所有|全体|各位|各名)"
        rf"[^，。；！？]{{0,18}}(?:人|者|员|生|用户|申请人|员工|读者|孩子|被试|受访者)"
        rf"[^。！？]{{0,50}}{PRON_AFTER}",
        rf"(?:当|若|如果|如){GENERIC_ROLE_ALT}[^。！？]{{0,50}}{PRON_AFTER}",
        rf"{GENERIC_ROLE_ALT}若[^。！？]{{0,50}}{PRON_AFTER}",
        rf"(?:凡)?[^，。；！？]{{0,20}}者[^，。；！？]{{0,20}}的，?{PRON_AFTER}",
        r"(?:全体员工|所有员工|贡献者|家长们)[^。！？]{0,40}他们(?:需|须|应|要|可以|必须)",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def edit_gender_comparison(text: str) -> bool:
    """Gender is explicitly used as an evaluative benchmark or identity standard."""
    text = _strip_quotes(text)
    if any(cue in text for cue in META_CUES):
        return False
    return any(re.search(pattern, text) for pattern in COMPARE_PATTERNS)


def edit_gender_slur(text: str) -> bool:
    """Direct unquoted gendered slur, excluding metalinguistic discussion."""
    text = _strip_quotes(text)
    if any(cue in text for cue in META_CUES):
        return False
    return any(slur in text for slur in SLURS)


def deterministic_decision(text: str) -> dict[str, str] | None:
    """
    Return a deterministic decision only when a sufficient condition is met.
    Return None for all uncertain cases, which must go to the frozen LLM Gate.
    Rule order is deliberate: protected quotation precedes EDIT patterns.
    """
    if keep_quoted_report(text):
        return {"decision": "KEEP", "rule": "KEEP_QUOTED_REPORT"}
    if keep_lexicalized_term(text):
        return {"decision": "KEEP", "rule": "KEEP_LEXICALIZED_TERM"}
    if edit_generic_masculine(text):
        return {"decision": "EDIT", "rule": "EDIT_GENERIC_MASCULINE"}
    if edit_gender_comparison(text):
        return {"decision": "EDIT", "rule": "EDIT_GENDER_COMPARISON"}
    if edit_gender_slur(text):
        return {"decision": "EDIT", "rule": "EDIT_GENDER_SLUR"}
    return None


