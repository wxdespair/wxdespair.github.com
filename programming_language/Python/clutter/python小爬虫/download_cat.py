import urllib.request
response = urllib.request.urlopen('http://placekitten.com/g/200/300')
"""上面的一句其实合并执行了两个操作
req = urllib.request.Request('http://placekitten.com/g/200/300')
response = urllib.request.urlopen(req)
"""
cat_img = response.read()
with open('cat_200_300.jpg', 'wb') as f:
	f.write(cat_img)

# 如果上述代码在idle中执行，则还可以使用一些其他的函数
>>> response.geturl()		# 返回网页URL
'http://placekitten.com/g/200/300'
>>> response.info()			# 返回目标服务器的一些相关信息
<http.client.HTTPMessage object at 0x04312110>
>>> print(response.info())	# 打印
Date: Thu, 03 Oct 2019 01:37:53 GMT
Content-Length: 10361
Connection: close
Set-Cookie: __cfduid=d3e5209bceb7fda368fd362c88a0e90a61570066673; expires=Fri, 02-Oct-20 01:37:53 GMT; path=/; domain=.placekitten.com; HttpOnly
Access-Control-Allow-Origin: *
Cache-Control: public, max-age=86400
Expires: Fri, 04 Oct 2019 01:37:53 GMT
CF-Cache-Status: HIT
Age: 25450
Accept-Ranges: bytes
Vary: Accept-Encoding
Server: cloudflare
CF-RAY: 51fb31830cba77c4-LAX


>>> response.getcode()		# 返回一个值，下面的200在这里代表正常访问
200