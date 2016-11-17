"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.9/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from trips.views import hello_world, home, search, search_form, uploadImg, showImg, train,adddata,showNewImg
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    url(r'^admin/',admin.site.urls),
    url(r'^hello/$',hello_world),
    url(r'^train/$',train),
    url(r'^adddata/$',adddata),
    url(r'^$',uploadImg),
    url(r'^search/$',search),
    url(r'^upload/$',uploadImg),
    url(r'^show/$',showImg),
    url(r'^shownew/$',showNewImg),
    url(r'^site_media/(?P<path>.*)', 'django.views.static.serve', {'document_root':'/home/kevin/facerc/pyface/yalefaces'}),
]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)+static(settings.STATIC_URL, document_root = settings.STATIC_ROOT )
