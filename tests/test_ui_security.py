"""UI security and accessibility tests.

These tests verify that the Flask application includes the expected security
headers on responses and that all rendered forms contain a hidden CSRF token
input. They also check that ARIA labels are present on key interactive
elements.
"""

import re
from bs4 import BeautifulSoup

from app import create_app


def get_client():
    """Create a Flask test client with the application factory."""
    app = create_app()
    app.testing = True
    return app.test_client()


def test_security_headers_present():
    client = get_client()
    resp = client.get('/')
    # Basic security headers added in app.after_request
    assert resp.headers.get('Content-Security-Policy') is not None
    assert resp.headers.get('Strict-Transport-Security') is not None
    assert resp.headers.get('X-Content-Type-Options') == 'nosniff'
    assert resp.headers.get('Referrer-Policy') == 'same-origin'
    assert resp.headers.get('Permissions-Policy') == 'geolocation=(), microphone=()'


def _assert_form_has_csrf(html: str):
    soup = BeautifulSoup(html, "html.parser")
    for form in soup.find_all('form'):
        hidden = form.find('input', {'type': 'hidden', 'name': 'csrf_token'})
        assert hidden is not None, f"Form missing CSRF token: {form}"


def test_public_auth_forms_use_accessible_csrf_wrappers():
    client = get_client()
    for path, label in [('/auth/login', 'Sign in')]:
        response = client.get(path)
        assert response.status_code == 200
        soup = BeautifulSoup(response.get_data(as_text=True), "html.parser")
        form = soup.find('form', {'aria-label': label})
        assert form is not None
        assert form.find('input', {'type': 'hidden', 'name': 'csrf_token'}) is not None


def test_login_form_contains_csrf():
    client = get_client()
    resp = client.get('/auth/login')
    assert resp.status_code == 200
    _assert_form_has_csrf(resp.get_data(as_text=True))


def test_admin_user_form_contains_csrf_and_aria():
    client = get_client()
    # Simple login; assumes an admin user exists with credentials admin/admin.
    login_page = client.get('/auth/login')
    csrf_token = BeautifulSoup(login_page.get_data(as_text=True), "html.parser").find('input', {'name': 'csrf_token'}).get('value')
    login_resp = client.post('/auth/login', data={
        'username': 'admin',
        'password': 'admin',
        'csrf_token': csrf_token,
    }, follow_redirects=True)
    assert login_resp.status_code == 200
    resp = client.get('/admin/users')
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    _assert_form_has_csrf(html)
    soup = BeautifulSoup(html, "html.parser")
    search_form = soup.find('form', {'aria-label': re.compile('search', re.I)})
    assert search_form is not None, "Search form missing aria-label"
