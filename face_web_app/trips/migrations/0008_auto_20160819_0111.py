# -*- coding: utf-8 -*-
# Generated by Django 1.9.8 on 2016-08-19 01:11
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trips', '0007_user'),
    ]

    operations = [
        migrations.CreateModel(
            name='Celebr',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=20)),
                ('content', models.TextField()),
                ('mem_id', models.IntegerField()),
                ('gender', models.TextField()),
                ('img', models.ImageField(upload_to='upload')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.DeleteModel(
            name='Post',
        ),
        migrations.DeleteModel(
            name='User',
        ),
        migrations.AlterField(
            model_name='img',
            name='img',
            field=models.ImageField(upload_to='temp'),
        ),
    ]
