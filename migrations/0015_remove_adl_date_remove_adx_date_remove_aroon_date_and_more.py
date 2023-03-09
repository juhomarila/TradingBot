# Generated by Django 4.1.5 on 2023-01-28 21:16

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('marketBotApp', '0014_alter_macd_macd_alter_macd_macd_signal'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='adl',
            name='date',
        ),
        migrations.RemoveField(
            model_name='adx',
            name='date',
        ),
        migrations.RemoveField(
            model_name='aroon',
            name='date',
        ),
        migrations.RemoveField(
            model_name='bbp8_model',
            name='date',
        ),
        migrations.RemoveField(
            model_name='macd',
            name='date',
        ),
        migrations.RemoveField(
            model_name='obv',
            name='date',
        ),
        migrations.RemoveField(
            model_name='optimal_buy_point',
            name='date',
        ),
        migrations.RemoveField(
            model_name='optimal_sell_point',
            name='date',
        ),
        migrations.RemoveField(
            model_name='rsi',
            name='date',
        ),
        migrations.RemoveField(
            model_name='sd',
            name='date',
        ),
        migrations.RemoveField(
            model_name='stock_1min',
            name='date',
        ),
        migrations.AddField(
            model_name='adl',
            name='stock',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='marketBotApp.finnish_stock_daily'),
        ),
        migrations.AddField(
            model_name='adx',
            name='stock',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='marketBotApp.finnish_stock_daily'),
        ),
        migrations.AddField(
            model_name='aroon',
            name='stock',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='marketBotApp.finnish_stock_daily'),
        ),
        migrations.AddField(
            model_name='bbp8_model',
            name='stock',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='marketBotApp.finnish_stock_daily'),
        ),
        migrations.AddField(
            model_name='macd',
            name='stock',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='marketBotApp.finnish_stock_daily'),
        ),
        migrations.AddField(
            model_name='obv',
            name='stock',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='marketBotApp.finnish_stock_daily'),
        ),
        migrations.AddField(
            model_name='optimal_buy_point',
            name='stock',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='marketBotApp.finnish_stock_daily'),
        ),
        migrations.AddField(
            model_name='optimal_sell_point',
            name='stock',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='marketBotApp.finnish_stock_daily'),
        ),
        migrations.AddField(
            model_name='rsi',
            name='stock',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='marketBotApp.finnish_stock_daily'),
        ),
        migrations.AddField(
            model_name='sd',
            name='stock',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='marketBotApp.finnish_stock_daily'),
        ),
        migrations.CreateModel(
            name='Signals',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('adl', models.FloatField()),
                ('obv', models.FloatField()),
                ('bbp8', models.FloatField()),
                ('adx', models.FloatField()),
                ('rsi', models.FloatField()),
                ('aroon_up', models.FloatField()),
                ('aroon_down', models.FloatField()),
                ('macd', models.FloatField()),
                ('macd_signal', models.FloatField()),
                ('std_dev', models.FloatField()),
                ('stock', models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='marketBotApp.finnish_stock_daily')),
            ],
        ),
    ]