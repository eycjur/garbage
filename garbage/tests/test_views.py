from django.test import LiveServerTestCase
from selenium.webdriver.chrome.webdriver import WebDriver


class TestClickSample(LiveServerTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.selenium = WebDriver(executable_path=r'garbage\tests\chromedriver.exe')

    @classmethod
    def tearDownClass(cls):
        cls.selenium.quit()
        super().tearDownClass()

    def test_open_page(self):
        """ページを開いてタイトルがあってるかを確認
        """
        self.selenium.get('http://localhost:8000/garbage')
        self.assertEquals("画像でゴミ分類！", self.selenium.title)





