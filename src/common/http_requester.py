#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""A module to provide a high level interface for HTTP requests."""

import requests

from common.string_util import StringUtil


class HttpRequester(object):
    """
    A class that contains high level methods to carry out HTTP calls. It supports TLS and
    setup authorization credentials in the form of a token.
    """

    def __init__(self, base_url, api_token=None, verify_cert=True):
        self._session = requests.Session()
        self._base_url = base_url
        self._verify_cert = verify_cert
        self._headers = {"Authorization": f"Token {api_token}"} if api_token else {}

    @property
    def webserver_api_path(self):
        """Return the configured web server that is used by default in the HTTP calls."""

        return self._base_url

    def _url(self, endpoint_sub_url):
        if "?" in endpoint_sub_url:
            return f"{self._base_url}{endpoint_sub_url}"
        else:
            return f"{self._base_url}{StringUtil.slash_suffix(endpoint_sub_url)}"

    def get(self, endpoint_sub_url, raw=False, **kwargs):
        """
        Implement a GET HTTP call.

        Parameters
        ----------
        endpoint_sub_url : str
            A relative path, starting from the configured web server route.
        raw : bool
            Whether to consider the `endpoint_sub_url` as a full URL or as a relative path.
        kwargs : dict
            A map of key-value pairs to be submitted to the GET operation.

        Returns
        -------
        requests.Response,
            The HTTP call response.
        """

        url = endpoint_sub_url if raw else self._url(endpoint_sub_url)
        return self._session.get(
            url, headers=self._headers.copy(), verify=self._verify_cert, **kwargs
        )

    def post(self, endpoint_sub_url, data=None, json=None, headers=None):
        """
        Implement a POST HTTP call.

        Parameters
        ----------
        endpoint_sub_url : str
            A relative path, starting from the configured web server route.
        data : (optional) dict, list of tuples, bytes, or file-like
            Object to send in the body of the request.
        json : (optional) dict,
            A json data to send in the body of the request.
        headers : (optional) dict
            Header attributes that will be set in the request.

        Returns
        -------
        requests.Response,
            The HTTP call response.
        """

        request_headers = self._headers.copy()
        if headers:
            request_headers.update(headers)

        url = self._url(endpoint_sub_url)
        return requests.post(
            url, data=data, json=json, headers=request_headers, verify=self._verify_cert
        )

    def patch(self, endpoint_sub_url, data=None, json=None, headers=None):
        """
        Implement a PATCH HTTP call.

        Parameters
        ----------
        endpoint_sub_url : str
            A relative path, starting from the configured web server route.
        data : (optional) dict, list of tuples, bytes, or file-like
            Object to send in the body of the request.
        json : (optional) dict,
            A json data to send in the body of the request.
        headers : (optional) dict
            Header attributes that will be set in the request.

        Returns
        -------
        requests.Response,
            The HTTP call response.
        """

        request_headers = self._headers.copy()
        if headers:
            request_headers.update(headers)

        url = self._url(endpoint_sub_url)
        return self._session.patch(
            url, data=data, json=json, headers=request_headers, verify=self._verify_cert
        )

    def delete(self, endpoint_sub_url):
        """
        Implement a DELETE HTTP call.

        Parameters
        ----------
        endpoint_sub_url : str
            A relative path, starting from the configured web server route.

        Returns
        -------
        requests.Response,
            The HTTP call response.
        """

        url = self._url(endpoint_sub_url)
        return self._session.delete(url, headers=self._headers.copy(), verify=self._verify_cert)
