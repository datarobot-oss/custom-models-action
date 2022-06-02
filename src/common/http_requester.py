import requests


class HttpRequester(object):
    def __init__(self, base_url, api_token=None):
        self._session = requests.Session()
        self._base_url = base_url
        self._headers = {"Authorization": f"Token {api_token}"} if api_token else {}

    def _url(self, endpoint_sub_url):
        if not endpoint_sub_url.endswith("/"):
            endpoint_sub_url += "/"
        return f"{self._base_url}/{endpoint_sub_url}"

    def get(self, endpoint_sub_url, raw=False, **kwargs):
        url = endpoint_sub_url if raw else self._url(endpoint_sub_url)
        return self._session.get(url, headers=self._headers.copy(), **kwargs)

    def post(self, endpoint_sub_url, data=None, json=None, headers=None):
        request_headers = self._headers.copy()
        if headers:
            request_headers.update(headers)

        url = self._url(endpoint_sub_url)
        return requests.post(url, data=data, json=json, headers=request_headers, verify=False)

    def patch(self, endpoint_sub_url, data=None, json=None, headers=None):
        request_headers = self._headers.copy()
        if headers:
            request_headers.update(headers)

        url = self._url(endpoint_sub_url)
        return self._session.patch(url, data=data, json=json, headers=request_headers, verify=False)

    def delete(self, endpoint_sub_url):
        url = self._url(endpoint_sub_url)
        return self._session.delete(url, headers=self._headers.copy(), verify=False)
