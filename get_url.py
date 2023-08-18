from selenium import webdriver
from time import sleep
from selenium.webdriver.common.keys import Keys
import csv
driver = webdriver.Chrome('/home/tima/detec_and_tracking/chromedriver_linux64/chromedriver')

driver.get('https://www.pexels.com/search/face/')
sleep(2)

# Scroll page foody to get more location
for _ in range(1000):
    driver.execute_script("window.scrollBy(0, 600)")
    sleep(2)

image_elements = driver.find_elements_by_tag_name("img")  # Tìm tất cả các thẻ <img>
   
image_urls = []  # Dùng để lưu trữ các đường dẫn ảnh


for image_element in image_elements:
    image_url = image_element.get_attribute("src")  # Trích xuất đường dẫn từ thuộc tính src
    if image_url:
        image_urls.append(image_url)
        
csv_filename = "image_urls.csv"
with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Image URLs"])
    csv_writer.writerows([[url] for url in image_urls])

print(f"Đã lưu {len(image_urls)} đường dẫn vào tệp {csv_filename}")
# # In tất cả các đường dẫn ảnh
# for url in image_urls:
#     print(url)

driver.quit()
