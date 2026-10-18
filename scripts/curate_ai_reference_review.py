"""Apply the source-checked Codex adjudication of the delegated AI review.

Gold excerpts become complete source lines/semantic units. The correction
recipe is explicit and replayable; no model suggestion is executed. Human
review records are preserved and never approved by this program.
"""

from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.real_dataset import LocatedEvidence, RealBenchmarkCase, validate_real_benchmark


def apply(root: Path):
    dev, test = validate_real_benchmark(root)
    cases = {case.id: case for case in [*dev.cases, *test.cases]}
    before = {key: value.model_dump(mode="json") for key, value in cases.items()}
    original_ids = list(cases)
    sources = {doc.corpus_path.removeprefix("corpus/"): (root / doc.corpus_path).read_text("utf-8")
               for doc in dev.manifest.documents}
    strict = "pydantic-v2--concepts--strict_mode.md"
    unions = "pydantic-v2--concepts--unions.md"
    fields = "pydantic-v2--concepts--fields.md"
    alias = "pydantic-v2--concepts--alias.md"
    models1 = "pydantic-v1--usage--models.md"
    models2 = "pydantic-v2--concepts--models.md"
    validators1 = "pydantic-v1--usage--validators.md"
    validators2 = "pydantic-v2--concepts--validators.md"
    serial1 = "pydantic-v1--usage--exporting_models.md"
    serial2 = "pydantic-v2--concepts--serialization.md"
    dataclasses = "pydantic-v2--concepts--dataclasses.md"
    adapter = "pydantic-v2--concepts--type_adapter.md"
    deps = "fastapi--en--docs--tutorial--dependencies--index.md"
    subdeps = "fastapi--en--docs--tutorial--dependencies--sub-dependencies.md"
    errors = "fastapi--en--docs--tutorial--handling-errors.md"
    status = "fastapi--en--docs--tutorial--response-status-code.md"
    response = "fastapi--en--docs--tutorial--response-model.md"
    direct = "fastapi--en--docs--advanced--response-directly.md"
    background = "fastapi--en--docs--tutorial--background-tasks.md"
    events = "fastapi--en--docs--advanced--events.md"
    security = "fastapi--en--docs--tutorial--security--first-steps.md"
    oauth = "fastapi--en--docs--tutorial--security--simple-oauth2.md"
    middleware = "fastapi--en--docs--tutorial--middleware.md"
    cors = "fastapi--en--docs--tutorial--cors.md"

    def quote(filename, marker, end_marker=None, *, exact=False):
        text = sources[filename]
        start = text.index(marker)
        last = start + len(marker)
        if end_marker:
            last = text.index(end_marker, start) + len(end_marker)
        first_line, last_line = text[:start].count("\n") + 1, text[:last].count("\n") + 1
        contents = marker if exact else "\n".join(text.splitlines()[first_line - 1:last_line]).strip()
        if len(contents) > 900:
            raise ValueError("split an overly broad gold passage into semantic evidence units")
        return LocatedEvidence(source_file=filename, contains=contents, start_line=first_line, end_line=last_line)

    def set_case(identifier, *, question=None, points=None, evidence=None):
        case = cases[identifier]
        if question:
            case.question = question
        if points:
            case.answer_key_points = points
        if evidence is not None:
            case.expected_evidence = evidence

    # Preserve neighboring qualifiers and the API labels of numeric defaults.
    for case in cases.values():
        case.expected_evidence = [quote(item.source_file, item.contains) for item in case.expected_evidence]

    set_case("REAL-DEV-001", evidence=[quote(strict, 'you can pass the string `"123"`'),
        quote(strict, "MyModel.model_validate({'x': '123'}, strict=True)"),
        quote(strict, "Input should be a valid integer [type=int_type")])
    set_case("REAL-DEV-002", question="在 Pydantic v2.8.2 中，smart union 的哪种匹配评分维度与严格模式直接相关？结合严格模式文档说明，为什么还需区分输入类型以及 Python/JSON 校验方式？", evidence=[
        quote(unions, "For `exactness`, Pydantic scores", "Validation would have succeeded in lax mode"),
        quote(strict, "For most types, when validating data from python"),
        quote(strict, "Note that we are looser", "but not from python:")])
    set_case("REAL-DEV-003", points=["从 Python 严格校验时，要求 UUID 实例，字符串输入不被接受。",
        "从 JSON 严格校验时，UUID 字符串仍可被接受并转换为 UUID。"], evidence=[
        quote(strict, "Note that we are looser", "but not from python:"),
        quote(strict, "Input should be an instance of UUID")])
    set_case("REAL-DEV-004", question="在 Pydantic v2.8.2 中，对按 int、str 顺序声明的联合类型校验字符串 '456'，left_to_right 和 smart 分别得到什么类型？两种结果为什么不矛盾？",
        points=["left_to_right 在默认宽松校验下先尝试 int，字符串 '456' 可转换为整数，因此返回 int 类型的 456。",
                "smart 对此类普通类型优先选择精确类型匹配，因此保留字符串 '456'；精确匹配成功后会立即返回，不会一律比较全部成员。",
                "差异来自所选联合校验模式的匹配策略，而不是同一模式下的矛盾。"], evidence=[
        quote(unions, "the first successful validation is accepted as input"),
        quote(unions, "id: Union[int, str] = Field(union_mode='left_to_right')"),
        quote(unions, "print(User(id='456'))", "#> id=456"),
        quote(unions, "An exact type match, for example"),
        quote(unions, "If validation succeeds with an exact type match")])
    set_case("REAL-DEV-007", question="In Pydantic v2.8.2, how can a first_name field accept both 'first_name' and 'fname' during validation, and will VSCode's initializer signature reflect a validation_alias? Explain runtime validation versus editor hints.",
        points=["Use Field(validation_alias=AliasChoices('first_name', 'fname')) to allow either input name.",
                "The fields guide says VSCode does not use validation_alias in the class initializer signature, even though Pydantic applies the alias at runtime."], evidence=[
        quote(alias, "`AliasChoices` is used to specify a choice of aliases."),
        quote(alias, "first_name: str = Field(validation_alias=AliasChoices('first_name', 'fname'))"),
        quote(fields, "Even though Pydantic treats `alias`", "class initializer signature.")])
    set_case("REAL-DEV-011", question="在 FastAPI 0.112.2 的依赖章节对 common_parameters 的文字说明中，列出了哪些可选查询参数及类型？skip、limit 的数值默认值分别是多少？",
        evidence=[quote(deps, "An optional query parameter `q`", "by default is `100`.")])
    set_case("REAL-DEV-014", points=["固定版本文档保留 Annotated 和 non-Annotated 示例，并建议尽可能使用 Annotated 写法。",
        "0.112.2 文档中同时展示的两种声明形式是可供选择的写法，不代表互相冲突的 API。"], evidence=[
        quote(deps, "Prefer to use the `Annotated` version if possible."),
        quote(deps, "FastAPI added support for `Annotated`"),
        quote(subdeps, "//// tab | Python 3.8+ non-Annotated")])
    cases["REAL-DEV-012"].expected_evidence.extend([
        quote(deps, "You only give `Depends` a single parameter."),
        quote(subdeps, "Notice that we are only declaring one dependency")])
    cases["REAL-DEV-013"].expected_evidence.append(quote(subdeps, "it will save the returned value in a"))
    set_case("REAL-DEV-009", points=["When alias and validation_alias are supplied together, validation_alias takes priority for validation.",
        "When alias and serialization_alias are supplied together, serialization_alias takes priority for serialization."], evidence=[
        quote(fields, "In case you use `alias` together", "over `alias` for serialization.")])
    set_case("REAL-DEV-017", evidence=[quote(status, "Notice that `status_code`"),
        quote(status, "It will:", "Document it as such in the OpenAPI schema"),
        quote(errors, "This also means that if you are inside a utility function"),
        quote(errors, "raise an exception with a status code of `404`")])
    set_case("REAL-DEV-018", evidence=[quote(errors, "**FastAPI** uses it so that, if you use a Pydantic model"),
        quote(errors, 'Instead, the client will receive an "Internal Server Error"'),
        quote(errors, "as that could expose a security vulnerability")])
    set_case("REAL-DEV-019", evidence=[quote(errors, "The only difference is that **FastAPI**'s `HTTPException`"),
        quote(errors, "you should register it for Starlette's `HTTPException`"),
        quote(errors, "This way, if any part of Starlette's internal code"),
        quote(errors, "from starlette.exceptions import HTTPException as StarletteHTTPException")])

    # Replace the dev-near-duplicate integer coercion question before any paid
    # held-out workflow run. The new fact is specific to model field tracking.
    set_case("REAL-TEST-001", question="在 pydantic v1.10.17 模型指南中，__fields_set__ 记录什么？基本用法示例中为什么结果是 {'id'}，而不包含有默认值的 name？",
        points=["__fields_set__ 记录初始化模型时明确设置过的字段名称。",
                "示例只显式传入 id，name 使用默认值，因此该集合为 {'id'}，不包含 name。"], evidence=[
        quote(models1, "`__fields_set__`", "Set of names of fields which were set"),
        quote(models1, "user = User(id='123')"),
        quote(models1, "assert user.__fields_set__ == {'id'}"),
        quote(models1, "`name` wasn't set when user was initialised")])
    set_case("REAL-TEST-002", points=["两个版本都强调对输出模型的类型和约束作保证，而不是保证原始输入数据本身符合这些类型。",
        "两个版本都用继承 BaseModel 的类来定义模型。"], evidence=[
        quote(models1, "In other words, *pydantic* guarantees"),
        quote(models2, "Pydantic guarantees the types and constraints of the output"),
        quote(models1, "(models are simply classes which inherit"),
        quote(models2, "Models are simply classes which inherit from", "[`pydantic.BaseModel`][pydantic.main.BaseModel]")])
    set_case("REAL-TEST-003", points=["对任意类实例或 ORM 对象，需设置 orm_mode=True，并通过 from_orm 构造模型实例。",
        "处理这些类实例时，可用 Config.getter_dict 指定自定义 GetterDict 子类，以定制字典式属性访问。"], evidence=[
        quote(models1, "The [Config](model_config.md) property `orm_mode`"),
        quote(models1, "The special constructor `from_orm`"),
        quote(models1, "Arbitrary classes are processed"),
        quote(models1, "provide a dictionary-like interface to any class."),
        quote(models1, "sub-class of `GetterDict` as the value of `Config.getter_dict`")])
    set_case("REAL-TEST-004", question="pydantic v2.8.2 中 .model_dump() 与 dict(user) 在嵌套模型上的行为有何差异？在文档的扁平模型示例中二者如何提供字段字典？", evidence=[
        quote(models2, "Either `.model_dump()` or `dict(user)`", "arguments."),
        quote(models2, "(Note that `dict(user)` will not recursively convert")])
    cases["REAL-TEST-004"].expected_evidence.append(quote(models2, "assert user.model_dump() == {'id': 123, 'name': 'Jane Doe'}"))
    set_case("REAL-TEST-006", evidence=[quote(validators1, 'validators are "class methods"', "of `UserModel`."),
        quote(validators1, "the second argument is always the field value"),
        quote(validators1, "you can also add any subset", "not explicitly listed in the signature"),
        quote(validators1, "validators should either return the parsed value")])
    set_case("REAL-TEST-007", evidence=[quote(validators1, "For performance reasons, by default validators", "to set a dynamic default value."),
        quote(validators1, "You'll often want to use this together with `pre`", "with `always=True`"),
        quote(validators2, "Validators won't run when the default value is used.", "where the function is called before the inner validator is called.")])
    set_case("REAL-TEST-008", evidence=[quote(validators2, "but only one `PlainValidator`", "will not call any inner validators."),
        quote(validators2, 'That is, it goes from right to left running all "before" validators')])
    set_case("REAL-TEST-009", points=["v1 describes validators as class methods whose first argument is the model class.",
        "v2 also describes field validators as class methods, recommending @classmethod below @field_validator for proper type checking."], evidence=[
        quote(validators1, 'validators are "class methods"', "of `UserModel`."),
        quote(validators2, '* `@field_validator`s are "class methods"')])
    set_case("REAL-TEST-011", points=["encoder 作为 default 参数传给 json.dumps()，默认使用内置自定义编码器。"],
        evidence=[quote(serial1, "`encoder`: a custom encoder")])
    set_case("REAL-TEST-012", evidence=[quote(serial1, "## `model.dict("),
        quote(serial1, "This is the primary way of converting a model"),
        quote(serial2, "## `model.model_dump("), quote(serial2, "This is the primary way of converting a model"),
        quote(serial1, "With this approach the raw field values", "returned, so sub-models will not be converted to dictionaries."),
        quote(serial2, "With this approach the raw field values", "sub-models will not be converted to dictionaries.")])
    set_case("REAL-TEST-013", question="在 pydantic v2.8.2 中，如何让一个 @field_serializer 作用于所有字段？另外，PlainSerializer 与 WrapSerializer 的 when_used 参数默认值是什么？", evidence=[
        quote(serial2, "A single serializer can also be called on all fields"),
        quote(serial2, "In addition, [`PlainSerializer`", "Both serializers accept optional arguments including:"),
        quote(serial2, "- `when_used`", "Defaults to 'always'.")])
    set_case("REAL-TEST-014", evidence=[quote(serial2, "With this approach the raw field values", "sub-models will not be converted to dictionaries."),
        quote(serial2, "Note also that [`RootModel`]"),
        quote(serial2, "The one exception to sub-models being converted", "done recursively.")])
    set_case("REAL-TEST-016", evidence=[quote(dataclasses, "arguments are the same as the standard decorator", "keyword argument `config` which has the same meaning")])
    set_case("REAL-TEST-017", question="In Pydantic v2.8.2, how should a standalone Pydantic dataclass be wrapped to validate data or generate JSON schema, and what adapter reuse policy does the TypeAdapter guide recommend for a loop?",
        points=["Wrap the dataclass with TypeAdapter and use the adapter's methods for validation or JSON schema generation.",
                "Create an adapter for a given type once and reuse it in loops: analyzing the type and constructing the core schema has non-trivial overhead."], evidence=[
        quote(dataclasses, "To perform validation or generate a JSON schema", "and make use of its methods."),
        quote(adapter, "When creating an instance of `TypeAdapter`", "just once and reuse it in loops or other performance-critical code.")])
    set_case("REAL-TEST-018", evidence=[quote(adapter, "`TypeAdapter`'s `dump_json` methods returns"),
        quote(adapter, "The reason for this discrepancy is that in V1"),
        quote(adapter, "Hence, for the new `TypeAdapter` class in V2")])
    set_case("REAL-TEST-019", evidence=[quote(dataclasses, "if you use the dataclass as a field of a `BaseModel`"),
        quote(dataclasses, "To perform validation or generate a JSON schema", "and make use of its methods.")])
    cases["REAL-TEST-019"].answer_key_points[1] = "For direct validation or JSON schema generation on a Pydantic dataclass, wrap it in TypeAdapter and use the adapter's methods."
    set_case("REAL-TEST-022", question="结合 FastAPI 0.112.2 的响应模型与直接返回响应文档，普通返回值默认如何转换并封装为响应？如何自动过滤密码等额外字段，直接返回 Response 又有何不同？",
        points=["普通返回值默认由 jsonable_encoder 转换成 JSON 兼容数据，再放入 JSONResponse。",
                "通过声明返回类型或 response_model 指定输出模型，FastAPI 会过滤未在输出模型中声明的字段。",
                "直接返回 Response 时 FastAPI 原样传递，不执行通常的转换和响应模型处理。"], evidence=[
        quote(direct, "By default, **FastAPI** would automatically convert"),
        quote(direct, "Then, behind the scenes, it would put that JSON-compatible data"),
        quote(response, "FastAPI will use this return type to:", "**Validate** the returned data."),
        quote(response, "FastAPI will use this `response_model` to do all"),
        quote(response, "filtering out all the data that is not declared in the output model"),
        quote(direct, "And when you return a `Response`, **FastAPI** will pass it directly.")])
    set_case("REAL-TEST-024", points=["并不冲突：常规返回数据根据声明的返回类型或响应模型进行校验，返回不合要求的数据代表应用错误。",
        "直接返回 Response 时 FastAPI 原样传递，该返回数据不会被自动校验、转换或生成对应文档。"], evidence=[
        quote(response, "* **Validate** the returned data."),
        quote(response, "If the data is invalid (e.g. you are missing a field)", exact=True),
        quote(response, "it means that *your* app code is broken, not returning what it should", exact=True),
        quote(response, "it will return a server error instead of returning incorrect data.", exact=True),
        quote(direct, "And when you return a `Response`, **FastAPI** will pass it directly."),
        quote(direct, "When you return a `Response` directly its data is not validated")])
    set_case("REAL-TEST-026", evidence=[quote(background, "The class `BackgroundTasks` comes directly from"),
        quote(background, "avoid accidentally importing the alternative `BackgroundTask`")])
    set_case("REAL-TEST-027", points=["Lifespan setup runs once before the application starts receiving requests; code after yield runs after handling requests, just before shutdown.",
        "Background tasks run after returning a response." ])
    set_case("REAL-TEST-029", points=["The recommended approach is lifespan; providing it means startup and shutdown event handlers are not called.",
        "The apparent conflict arises because the guide also documents the deprecated event-based alternative: choose lifespan or events, not both."], evidence=[
        quote(events, "## Alternative Events (deprecated)"),
        quote(events, "The recommended way to handle the *startup*")])
    set_case("REAL-TEST-031", points=["直接返回 401 UNAUTHORIZED 错误，无需另行检查 token 是否存在。"],
        evidence=[quote(security, "If it doesn't see an `Authorization` header")])
    set_case("REAL-TEST-032", evidence=[quote(security, "When we create an instance of the `OAuth2PasswordBearer`"),
        quote(security, "This parameter doesn't create that endpoint"),
        quote(oauth, "The spec also states that the `username` and `password`"),
        quote(oauth, "And the spec says that the fields have to be named like that."),
        quote(security, "it sends a header `Authorization` with a value of `Bearer `")])
    cases["REAL-TEST-033"].answer_key_points[1] = "改用 OAuth2PasswordRequestFormStrict 后，会强制要求 grant_type 为规范指定的固定值 password。"
    set_case("REAL-TEST-034", evidence=[quote(oauth, "The form field name is `scope`"),
        quote(oauth, "The instance of the dependency class `OAuth2PasswordRequestForm` won't have")])
    cases["REAL-TEST-034"].answer_key_points = ["输入表单字段 scope（单数）是可选的、以空格分隔的字符串。",
        "OAuth2PasswordRequestForm 接收该输入后，在实例的 scopes 属性上提供拆分后的字符串列表，因此输入形式与解析后的属性并不冲突。"]
    cases["REAL-TEST-034"].expected_evidence.append(quote(oauth, "An optional `scope` field"))
    set_case("REAL-TEST-036", evidence=[quote(middleware, "If you have dependencies with `yield`"),
        quote(middleware, "If there were any background tasks")])
    set_case("REAL-TEST-037", points=["Import CORSMiddleware, create the allowed-origins list, and add the middleware to the FastAPI application.",
        "To expose custom response headers to browser clients, configure the CORS expose_headers parameter."], evidence=[
        quote(middleware, "using the parameter `expose_headers`", exact=True),
        quote(middleware, "you need to add them to your CORS configurations", exact=True),
        quote(cors, "* Import `CORSMiddleware`."), quote(cors, "* Create a list of allowed origins"),
        quote(cors, '* Add it as a "middleware" to your **FastAPI** application.')])
    set_case("REAL-TEST-038", evidence=[quote(cors, "* `allow_methods`"), quote(cors, "* `allow_headers`"), quote(cors, "* `max_age`")])
    set_case("REAL-TEST-039", evidence=[quote(cors, "excluding everything that involves credentials"), quote(cors, "* `allow_credentials`")])

    encoded = {}
    changes = []
    previous_path = root / "review/corrections.json"
    previous = {entry["id"]: entry for entry in json.loads(previous_path.read_text("utf-8"))["changes"]} if previous_path.exists() else {}
    for identifier in original_ids:
        case = RealBenchmarkCase.model_validate(cases[identifier].model_dump())
        unique = {(e.source_file, e.contains): e for e in case.expected_evidence}
        case.expected_evidence = [e for e in unique.values() if not any(
            e.source_file == other.source_file and e.contains != other.contains and e.contains in other.contains
            for other in unique.values())]
        if case.human_review.model_dump(mode="json") != before[identifier]["human_review"]:
            raise ValueError("AI curation must not change human review")
        original = previous[identifier]["before"] if identifier in previous else before[identifier]
        after = case.model_dump(mode="json")
        fields_changed = [name for name in after if original[name] != after[name]]
        if fields_changed:
            changes.append(dict(id=identifier, fields_changed=fields_changed, before=original, after=after))
        encoded.setdefault(case.split, []).append(case.model_dump_json())
    for split, lines in encoded.items():
        (root / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    validate_real_benchmark(root)
    if changes:
        (root / "review/corrections.json").write_text(json.dumps(dict(
            schema_version="ai-reference-corrections-v1", adjudicator="Codex (AI)", changes=changes,
        ), ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Source-checked changes to {len(changes)} cases; human approval unchanged.")


if __name__ == "__main__":
    apply(ROOT / "benchmarks/real-v1")
