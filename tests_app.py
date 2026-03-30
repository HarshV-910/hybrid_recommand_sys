import requests
import time
# to test that is streamlit app opening or not
app_url = "http://localhost:8000"

def get_status(url):
    response = requests.get(url)
    return response.status_code

def test_app():
    time.sleep(60)
    status_code = get_status(app_url) 
    assert status_code == 200, f"Unable to load Streamlit app, status code: {status_code}"
        