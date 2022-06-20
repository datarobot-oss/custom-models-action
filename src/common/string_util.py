class StringUtil:
    @staticmethod
    def slash_suffix(s):
        return s if s.endswith("/") else f"{s}/"
