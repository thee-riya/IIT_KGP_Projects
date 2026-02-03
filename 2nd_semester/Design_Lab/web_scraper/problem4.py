from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Initialize WebDriver
driver = webdriver.Firefox()
wait = WebDriverWait(driver, 10)

# Navigate to Amazon Today's Deals page
driver.get('https://www.amazon.in/')

try:
    # Click "Today's Deals"
    link = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "Today's Deals")))
    link.click()

    parent_div = wait.until(EC.presence_of_element_located((By.ID, "DealsGridScrollAnchor")))
    carousel = parent_div.find_element(By.XPATH, ".//div[contains(@class, 'a-carousel-container')]")
    li = carousel.find_elements(By.XPATH, ".//li[contains(@class, 'a-carousel-card')]")

    
    print(len(li))
    # categories = []
    for i in li:
        buttons = i.find_elements(By.TAG_NAME, "button")  # List of buttons inside `li`
        for button in buttons:
            try:
                category_name = button.text.strip()  # Extract text from button
                button.click()
                time.sleep(2)
                print("Category:", category_name)
            except Exception as e:
                print("Error:", e)

except Exception as e:
    print("Error:", e)

driver.quit()