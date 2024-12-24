import os
import time
import requests
from selenium import webdriver
from selenium import webdriver

# Define the URL to scrape
WEB_URL = "https://lens.google.com/search?ep=gsbubb&hl=fr&re=df&p=AbrfA8ovn3bn4smlz19CoAW2xy94wjmiH3VxeqD9qI3y7sLJCfMrsBwchBHM9Fk0oBlgL5ua4mJshnJFiJruEJv0nuYx8UzyYXzB0QthKA5HTxp8zr2X9siI60GFNnDuJ308dPxi_941sQh4UIsaEsXysbG01F5NmvSc4M7lLnNZmVfGCbmOWJlRs_rHOIhon-B6cqEdiPf9b4QJqA%3D%3D#lns=W251bGwsbnVsbCxudWxsLG51bGwsbnVsbCxudWxsLG51bGwsIkVrY0tKREUwTkdaa056bG1MV0l3TXpFdE5ETTNOQzFpTkRZeUxXRmxOVEJsWVRrNE5qaG1OeElmY3psb1VVWkpkbFl6Wkc5Uk5FWXpjRVF5ZGpoaFkzWXRMVmxmZFVwU2F3PT0iLG51bGwsbnVsbCxudWxsLG51bGwsbnVsbCxudWxsLG51bGwsWyJkODJkZjQ1NS1hYjk2LTRkYzgtYjVkNi02ZGY1MTgwOWJjYTUiXV0="

# Initialize a Firefox webdriver
# Initialize a Chromium webdriver
driver = webdriver.Chrome()

# Open the webpage
driver.get(WEB_URL)

# Wait for 5 seconds to let the page load
time.sleep(5)

# Execute JavaScript to get src attributes of specific elements
elements = driver.execute_script("""
    var elements = document.getElementsByClassName("wETe9b jFVN1");
    var srcList = [];
    for (let element of elements) {
        srcList.push(element.src);
    }
    return srcList; 
""")

# Close the webdriver
driver.quit()

# Create directory to save downloaded images if it doesn't exist
if not os.path.exists('downloaded_images'):
    os.makedirs('downloaded_images')

# Download each image
for index, image_url in enumerate(elements):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            # Save the image to disk
            with open(f"downloaded_images/image_{index}.jpg", 'wb') as f:
                f.write(response.content)
            print(f"Image {index} downloaded successfully.")
        else:
            print(f"Failed to download image {index}: HTTP status code {response.status_code}")
    except Exception as e:
        print(f"Error downloading image {index}: {str(e)}")

print("All images downloaded.")