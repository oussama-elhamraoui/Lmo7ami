from django.urls import path
from . import views

urlpatterns = [
    path('', views.test),
    path('send', views.send_message),
    path('signup', views.signup),
    path('login', views.login),
    path('new', views.new_conversation)
]
