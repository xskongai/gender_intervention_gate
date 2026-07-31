from gender_gate.cleaning import audit_negative, audit_positive


def test_positive_hidden_context_is_reviewed():
    row = {
        "编号": "POS-X",
        "输入句子": "王某某先生台鉴：现致函如下。",
        "L1类别": "一、缺省预设（信息缺位处默认为男性）",
        "L2子类": "1.6 收件与受众称谓默认",
        "问题说明": "收件人性别未经核实。",
        "备注": "",
        "处置理由": "",
        "是否争议": "否",
        "严重度": "应改",
        "难度": "中",
        "语体大类": "法律与司法",
        "语体": "律师函",
        "切分": "dev",
    }
    result = audit_positive(row)
    assert result.priority == "HIGH"
    assert result.suggested_action == "MOVE_TO_NEGATIVE"


def test_positive_mixed_item_is_delete_or_split():
    row = {
        "编号": "POS-Y",
        "输入句子": "男款跑鞋 / 女款跑鞋 / 女生减脂专区",
        "L1类别": "二、冗余标记（性别在场却不承载信息）",
        "L2子类": "2.6 商品、空间与内容的性别化归类",
        "问题说明": "",
        "备注": "",
        "处置理由": "",
        "是否争议": "是",
        "严重度": "应改",
        "难度": "中",
        "语体大类": "商业与产品",
        "语体": "电商类目树",
        "切分": "dev",
    }
    result = audit_positive(row)
    assert result.priority == "HIGH"
    assert result.suggested_action == "DELETE_OR_SPLIT"


def test_reviewed_negative_is_not_requeued():
    row = {
        "编号": "NEG-X",
        "输入句子": "一会儿麻醉师会过来，他会说明风险。",
        "L1类别": "一、指称已知性别的具体个人",
        "L2子类": "1.2 人称代词",
        "保留理由": "单句证据不足。",
        "备注": "",
        "处置理由": "单句证据不足，优先保持原文",
        "是否争议": "否",
        "难度": "中",
        "语体大类": "医疗健康",
        "语体": "导诊话术",
        "切分": "dev",
    }
    result = audit_negative(row)
    assert result.priority == "PASS"
    assert result.suggested_action == "KEEP_CURRENT"
