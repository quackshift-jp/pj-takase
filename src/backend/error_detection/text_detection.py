def detect_text_error(
    ocr_data: dict[str, any], accrate_data: dict[str, any]
) -> dict[str, str]:
    """業者記入テキストとOCRを比較して、エラー検知を行う
    args:
    ocr_data dict[str,any]: OCRのテキスト
    ex)
        {
            "物件名": "物件A",
            "管理会社": "会社A",
            "賃料": 111,
            "退去条件": "退去条件A",
        }

    accrate_data dict[str,any]: 比較するテキスト(業者記入)
    ex)
        {
            "物件名": "物件B",
            "管理会社": "会社B",
            "賃料": 222,
            "退去条件": "退去条件B",
        }

    return dict[str,any]: エラー検知結果を報告
    ex)
        {
            "物件名": "結果",
            "管理会社": "結果",
            "賃料": "Warning: {正しい内容}ではないですか？",
            "退去条件":  "Warning: {正しい内容}ではないですか？",
        }
    """
    error_detection = {}
    for key in ocr_data.keys():
        if ocr_data[key] != accrate_data[key]:
            error_detection[key] = f"{accrate_data[key]}ではないですか？"
        else:
            error_detection[key] = "正しく記入できています"
    # TODO:長文の場合はエラーハンドリングも長くなってしまうので、処理方法を考える
    return error_detection
