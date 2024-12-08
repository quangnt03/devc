import re
def extract_folder_id(url):
    match = re.search(r"[-\w]{25,}", url)
    if match:
        return match.group(0)
    return None
