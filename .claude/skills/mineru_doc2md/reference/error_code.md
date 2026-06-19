| Error Code | Description | Solution |
| ---------- | ----------  | -------- |
| A0202 | Invalid Token | Check if the Token is correct, verify the Bearer prefix, or replace with a new Token |
| A0211 | Token expired | Replace with a new Token |
| -500 | Parameter error | Ensure parameter types and Content-Type are correct |
| -10001 | Service error | Please try again later |
| -10002 | Request parameter error | Check request parameter format |
| -60001 | Failed to generate upload URL | Please try again later |
| -60002 | Failed to match file format | File type detection failed. Ensure the file name and URL contain the correct extension, and the file is one of: pdf, doc, docx, ppt, pptx, xls, xlsx, png, jp(e)g |
| -60003 | File read failure | Check if the file is corrupted and re-upload |
| -60004 | Empty file | Please upload a valid file |
| -60005 | File size exceeds limit | Check file size, maximum supported is 200MB |
| -60006 | Page count exceeds limit | Please split the file and try again |
| -60007 | Model service temporarily unavailable | Please try again later or contact technical support |
| -60008 | File read timeout | Check that the URL is accessible |
| -60009 | Task submission queue is full | Please try again later |
| -60010 | Extract failed | Please try again later |
| -60011 | Failed to get valid file | Please ensure the file has been uploaded |
| -60012 | Task not found | Please ensure the task_id is valid and not deleted |
| -60013 | No permission to access this task | You can only access tasks you submitted |
| -60014 | Deleting a running task | Running tasks cannot be deleted at this time |
| -60015 | File conversion failed | You can manually convert to PDF and re-upload |
| -60016 | File conversion failed | Failed to convert file to the specified format. Try exporting in another format or retry |
| -60017 | Retry limit reached | Wait for future model upgrades and try again |
| -60018 | Daily extract task limit reached | Please try again tomorrow |
| -60019 | HTML file extract quota exhausted | Please try again tomorrow |
| -60020 | File splitting failed | Please try again later |
| -60021 | Failed to read page count | Please try again later |
| -60022 | Web page read failure | May be caused by network issues or rate limiting. Please try again later |
