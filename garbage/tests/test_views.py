from django.test import LiveServerTestCase
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By


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

    def test_click_sample(self):
        """モデルの動きが正常かを確認
        サンプル画像をクリックした際の挙動を確認
        """
        self.selenium.get('http://localhost:8000/garbage')
        self.selenium.find_elements_by_class_name("sample-img")[1].click()

        num_tr = len(self.selenium.find_elements(By.XPATH, '//tr'))
        self.assertEquals(6, num_tr)

