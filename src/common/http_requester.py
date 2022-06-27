import requests

from common.string_util import StringUtil


class HttpRequester(object):
    def __init__(self, base_url, api_token=None, verify_cert=True):
        self._session = requests.Session()
        self._base_url = base_url
        self._verify_cert = verify_cert
        self._headers = {"Authorization": f"Token {api_token}"} if api_token else {}

    @property
    def webserver_api_path(self):
        return self._base_url

    def _url(self, endpoint_sub_url):
        return f"{self._base_url}{StringUtil.slash_suffix(endpoint_sub_url)}"

    def get(self, endpoint_sub_url, raw=False, **kwargs):
        url = endpoint_sub_url if raw else self._url(endpoint_sub_url)
        return self._session.get(
            url, headers=self._headers.copy(), verify=self._verify_cert, **kwargs
        )

    def post(self, endpoint_sub_url, data=None, json=None, headers=None):
        request_headers = self._headers.copy()
        if headers:
            request_headers.update(headers)

        url = self._url(endpoint_sub_url)
        return requests.post(
            url, data=data, json=json, headers=request_headers, verify=self._verify_cert
        )

    def patch(self, endpoint_sub_url, data=None, json=None, headers=None):
        request_headers = self._headers.copy()
        if headers:
            request_headers.update(headers)

        url = self._url(endpoint_sub_url)
        return self._session.patch(
            url, data=data, json=json, headers=request_headers, verify=self._verify_cert
        )

    def delete(self, endpoint_sub_url):
        url = self._url(endpoint_sub_url)
        return self._session.delete(url, headers=self._headers.copy(), verify=self._verify_cert)
