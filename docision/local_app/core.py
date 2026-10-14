"""Pure input/output helpers for the local application."""
import csv
import io

TEMPLATES = {
    '文件分类': ('这一页属于哪种文档？', ['发票', '合同', '申请表', '说明书', '其他', '无法判断']),
    '费用分类': ('这张票据对应哪类支出？', ['住宿', '交通', '餐饮', '其他', '无法判断']),
    '签名检查': ('文档的签名栏是否有可见签名？', ['有', '无', '无法判断']),
    '盖章检查': ('文档的盖章区域是否有可见印章？', ['有', '无', '无法判断']),
    '表单漏填检查': ('联系电话栏是否填写？', ['已填写', '未填写', '无法判断']),
    '表格页筛选': ('这一页是否包含表格？', ['包含', '不包含', '无法判断']),
    '扫描质量检查': ('这一页是否存在影响阅读的明显模糊？', ['是', '否', '无法判断']),
    '图纸分类': ('这张图纸属于哪一类？', ['建筑平面图', '机械零件图', '电气原理图', '其他', '无法判断']),
    '自定义判断': ('', ['是', '否', '无法判断']),
}

def question_spec(question, candidates):
    if not isinstance(question, str) or not question.strip():
        raise ValueError('请输入问题。')
    if len(question) > 8000:
        raise ValueError('问题过长，请缩短后重试。')
    if not isinstance(candidates, list) or not all(isinstance(x, str) for x in candidates):
        raise ValueError('候选答案必须是文本列表。')
    choices = [x.strip() for x in candidates if x.strip()]
    if not 2 <= len(choices) <= 26 or len(set(choices)) != len(choices):
        raise ValueError('请提供 2–26 个不重复的候选答案，每行一个。')
    if any(len(x) > 2000 for x in choices):
        raise ValueError('候选答案过长。')
    return {'id': 'q1', 'question': question.strip(),
            'candidates': {chr(65+i): x for i, x in enumerate(choices)}}

MAX_QUESTIONS = 20


def question_specs(body):
    """Normalize a question list; retain the original single-question API."""
    questions = body.get('questions') if 'questions' in body else [body]
    if not isinstance(questions, list) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ValueError(f'请提供 1–{MAX_QUESTIONS} 个问题。')
    specs = []
    for index, item in enumerate(questions, 1):
        if not isinstance(item, dict):
            raise ValueError(f'问题 {index} 必须包含问题文本和候选答案。')
        try:
            spec = question_spec(item.get('question'), item.get('candidates'))
        except ValueError as error:
            raise ValueError(f'问题 {index}：{error}') from error
        spec['id'] = f'q{index}'
        specs.append(spec)
    return specs


def csv_export(rows):
    output = io.StringIO(newline='')
    writer = csv.writer(output)
    writer.writerow(['文件', '页码', '问题编号', '问题', '候选答案', '模型判断', '人工修正', '候选概率', '错误'])
    def safe(value):
        value = str(value or '')
        return "'" + value if value.lstrip().startswith(('=', '+', '-', '@')) else value
    for row in rows:
        result = row.get('result', {})
        writer.writerow([safe(row.get('file')), row.get('page'), safe(row.get('question_id')), safe(row.get('question')),
                         safe(' / '.join(row.get('candidates', []))),
                         safe(result.get('decision', {}).get('text')), safe(row.get('correction')),
                         safe('; '.join(f"{c['text']}: {c['probability']:.6f}" for c in result.get('candidates', []))),
                         safe(row.get('error'))])
    return '\ufeff' + output.getvalue()
