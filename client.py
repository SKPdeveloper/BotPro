#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Клієнтська частина Telegram-бота для автоматичного наповнення та просування каналу.
"""

import os
import sys
import json
import time
import logging
import asyncio
import sqlite3
import datetime
import traceback
from io import BytesIO
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("client.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Конфігураційні змінні з наданими даними
BOT_TOKEN = "7581835062:AAF0o7fJ1LwfGYYrQWo5E7t-GqYtXJPcxsM"
DATABASE_PATH = "tg_bot.db"
BOT_NAME = "PosterBot"

# Відображення інформації про конфігурацію
logger.info("------------------------")
logger.info("Конфігурація клієнта:")
logger.info(f"BOT_TOKEN: {BOT_TOKEN}")
logger.info(f"BOT_NAME: {BOT_NAME}")
logger.info("------------------------")

class TelegramClient:
    """Клас для взаємодії з користувачем через Telegram Bot API"""
    
    def __init__(self, token):
        """
        Ініціалізація клієнтської частини
        
        :param token: Токен Telegram бота
        """
        self.token = token
        self.application = None
        self.db_conn = None
        
        logger.info("Ініціалізація клієнтської частини бота")
        
        # Ініціалізація бази даних
        self._init_database()
    
    def _init_database(self):
        """Ініціалізація підключення до бази даних"""
        logger.info("Підключення до бази даних")
        
        try:
            if not os.path.exists(DATABASE_PATH):
                logger.error("База даних не знайдена. Переконайтеся, що серверна частина запущена.")
                return
            
            self.db_conn = sqlite3.connect(DATABASE_PATH)
            logger.info("Успішне підключення до бази даних")
            
        except Exception as e:
            logger.error(f"Помилка при підключенні до бази даних: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробник команди /start"""
        logger.info(f"Отримано команду /start від користувача {update.effective_user.id}")
        
        # Створення головного меню
        keyboard = [
            [InlineKeyboardButton("📊 Статистика каналу", callback_data="stats")],
            [InlineKeyboardButton("📝 Запланувати пост", callback_data="schedule")],
            [InlineKeyboardButton("🔍 Аналіз аудиторії", callback_data="audience")],
            [InlineKeyboardButton("😀 Аналіз емодзі", callback_data="emoji")],
            [InlineKeyboardButton("📈 Аналіз постів", callback_data="posts")],
            [InlineKeyboardButton("⏰ Оптимальний час", callback_data="optimal_time")],
            [InlineKeyboardButton("🌐 Пошук релевантних каналів", callback_data="search_channels")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"👋 Вітаю! Я {BOT_NAME} - бот для автоматичного наповнення та просування вашого Telegram-каналу.\n\n"
            "Оберіть дію з меню нижче:",
            reply_markup=reply_markup
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробник команди /help"""
        logger.info(f"Отримано команду /help від користувача {update.effective_user.id}")
        
        help_text = (
            f"📚 *Довідка по використанню бота {BOT_NAME}*\n\n"
            "*Основні команди:*\n"
            "/start - Головне меню\n"
            "/help - Показати цю довідку\n"
            "/stats - Статистика каналу\n"
            "/schedule - Запланувати новий пост\n"
            "/audience - Аналіз аудиторії\n"
            "/emoji - Аналіз емодзі-реакцій\n"
            "/posts - Аналіз ефективності постів\n"
            "/optimal_time - Оптимальний час для публікацій\n"
            "/search - Пошук релевантних каналів\n\n"
            "*Використання функцій автопостингу:*\n"
            "1. Використовуйте команду /schedule\n"
            "2. Введіть текст публікації\n"
            "3. Прикріпіть медіа-файл (опційно)\n"
            "4. Вкажіть дату та час публікації\n\n"
            "*Аналітика:*\n"
            "Бот автоматично збирає дані про ваш канал. "
            "Використовуйте відповідні команди для перегляду різних типів аналітики."
        )
        
        await update.message.reply_text(
            help_text,
            parse_mode='Markdown'
        )
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробник натискань на кнопки меню"""
        query = update.callback_query
        await query.answer()
    
        logger.info(f"Отримано натискання на кнопку: {query.data} від користувача {query.from_user.id}")
    
        try:
            if query.data == "stats":
                logger.info("Перехід до статистики каналу")
                await self.show_channel_stats(query, context)
            elif query.data == "schedule":
                logger.info("Перехід до планування поста")
                await self.schedule_post_step1(query, context)
            elif query.data == "audience":
                logger.info("Перехід до аналізу аудиторії")
                await self.show_audience_analysis(query, context)
            elif query.data == "emoji":
                logger.info("Перехід до аналізу емодзі")
                await self.show_emoji_analysis(query, context)
            elif query.data == "posts":
                logger.info("Перехід до аналізу постів")
                await self.show_posts_analysis(query, context)
            elif query.data == "optimal_time":
                logger.info("Перехід до оптимального часу")
                await self.show_optimal_time(query, context)
            elif query.data == "search_channels":
                logger.info("Перехід до пошуку каналів")
                await self.search_channels_step1(query, context)
            elif query.data == "back_to_main":
                logger.info("Повернення до головного меню")
                await self.back_to_main_menu(query, context)
            elif query.data.startswith("day_"):
                # Обробка вибору дня тижня для планування поста
                day = query.data.split("_")[1]
                logger.info(f"Вибрано день: {day}")
                context.user_data["schedule_day"] = day
                await self.schedule_post_step3(query, context)
            elif query.data.startswith("hour_"):
                # Обробка вибору години для планування поста
                hour = query.data.split("_")[1]
                logger.info(f"Вибрано годину: {hour}")
                context.user_data["schedule_hour"] = hour
                await self.schedule_post_step4(query, context)
            elif query.data.startswith("minute_"):
                # Обробка вибору хвилин для планування поста
                minute = query.data.split("_")[1]
                logger.info(f"Вибрано хвилини: {minute}")
                context.user_data["schedule_minute"] = minute
                await self.schedule_post_confirm(query, context)
            elif query.data == "add_media":
                # Перехід в режим очікування медіа-файлу
                logger.info("Обробка кнопки 'Додати медіа'")
                context.user_data['waiting_for'] = 'post_media'
                await query.edit_message_text(
                    "📎 Надішліть медіа-файл (фото, відео або документ) для публікації.\n\n"
                    "Якщо ви передумали додавати медіа, введіть /skip для продовження без медіа-файлу."
                )
            elif query.data == "no_media":
                # Перехід до вибору часу публікації без медіа
                logger.info("Обробка кнопки 'Ні, тільки текст'")
                await self.schedule_post_step2(query, context)
            elif query.data == "confirm_post":
                logger.info("Підтвердження публікації поста")
                await self.confirm_post(query, context)
            elif query.data == "cancel_post":
                logger.info("Скасування публікації поста")
                await self.cancel_post(query, context)
            else:
                logger.warning(f"Отримано невідому кнопку: {query.data}")
                await query.edit_message_text(
                    f"🤔 Невідома команда: {query.data}\nБудь ласка, поверніться до головного меню.",
                    reply_markup=self.get_back_button()
                )
        except Exception as e:
            logger.error(f"Помилка при обробці кнопки {query.data}: {str(e)}")
            logger.error(traceback.format_exc())
            try:
                await query.edit_message_text(
                    f"❌ Сталася помилка при обробці команди: {str(e)}",
                    reply_markup=self.get_back_button()
                )
            except Exception as edit_error:
                logger.error(f"Не вдалося відредагувати повідомлення: {str(edit_error)}")
                try:
                    await query.message.reply_text(
                        f"❌ Сталася помилка при обробці команди: {str(e)}",
                        reply_markup=self.get_back_button()
                    )
                except Exception as reply_error:
                    logger.error(f"Не вдалося відправити повідомлення: {str(reply_error)}")
    
    async def show_audience_analysis(self, query, context):
        """Показати аналіз аудиторії"""
        logger.info("Відображення аналізу аудиторії")
        
        await query.edit_message_text(
            "⏳ Завантаження аналізу аудиторії...",
            reply_markup=None
        )
        
        try:
            # Отримання даних про аудиторію
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT date, subscribers FROM audience
                ORDER BY date ASC
            """)
            
            audience_data = cursor.fetchall()
            
            if not audience_data:
                await query.edit_message_text(
                    "❌ Немає даних про аудиторію. Переконайтеся, що серверна частина запущена та налаштована правильно.",
                    reply_markup=self.get_back_button()
                )
                return
            
            # Створення графіку динаміки аудиторії
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import pandas as pd
            
            df = pd.DataFrame(audience_data, columns=['date', 'subscribers'])
            df['date'] = pd.to_datetime(df['date'])
            df['growth'] = df['subscribers'].diff()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
            
            # Графік кількості підписників
            ax1.plot(df['date'], df['subscribers'], 'b-', marker='o')
            ax1.set_title('Динаміка аудиторії каналу')
            ax1.set_ylabel('Кількість підписників')
            ax1.grid(True)
            
            # Графік приросту
            ax2.bar(df['date'][1:], df['growth'][1:], color=['g' if x >= 0 else 'r' for x in df['growth'][1:]])
            ax2.set_title('Щоденний приріст підписників')
            ax2.set_ylabel('Приріст')
            ax2.grid(True)
            
            # Форматування дат на осі x
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
                if len(df) > 10:
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=len(df) // 10))
                plt.setp(ax.get_xticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Збереження графіку у буфер пам'яті
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Формування статистичної інформації
            current_subscribers = df['subscribers'].iloc[-1]
            first_subscribers = df['subscribers'].iloc[0]
            total_growth = current_subscribers - first_subscribers
            total_growth_percent = (total_growth / first_subscribers) * 100 if first_subscribers > 0 else 0
            
            avg_daily_growth = df['growth'][1:].mean() if len(df) > 1 else 0
            max_daily_growth = df['growth'][1:].max() if len(df) > 1 else 0
            
            # Прогноз росту
            if len(df) >= 7:  # Якщо є дані хоча б за тиждень
                weekly_growth_rate = df['subscribers'].iloc[-1] / df['subscribers'].iloc[-7] if df['subscribers'].iloc[-7] > 0 else 1
                monthly_projection = current_subscribers * (weekly_growth_rate ** 4)
                projection_text = f"📈 Прогноз на місяць: ~{int(monthly_projection)} підписників"
            else:
                projection_text = "📊 Недостатньо даних для прогнозу"
            
            stats_text = (
                f"👥 *Аналіз аудиторії каналу*\n\n"
                f"👥 Поточна кількість підписників: {current_subscribers}\n"
                f"📈 Загальний приріст: {total_growth:+d} ({total_growth_percent:.1f}%)\n"
                f"📊 Середній щоденний приріст: {avg_daily_growth:.1f}\n"
                f"🚀 Максимальний денний приріст: {max_daily_growth:.1f}\n"
                f"{projection_text}\n\n"
                f"📅 Період аналізу: {df['date'].min().strftime('%d.%m.%Y')} - {df['date'].max().strftime('%d.%m.%Y')}"
            )
            
            # Відправлення графіку
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=buf,
                caption=stats_text,
                parse_mode='Markdown',
                reply_markup=self.get_back_button()
            )
            
            # Редагування оригінального повідомлення
            await query.edit_message_text(
                "✅ Аналіз аудиторії каналу згенеровано успішно.",
                reply_markup=self.get_back_button()
            )
            
        except Exception as e:
            logger.error(f"Помилка при відображенні аналізу аудиторії: {str(e)}")
            logger.error(traceback.format_exc())
            
            await query.edit_message_text(
                f"❌ Помилка при аналізі аудиторії: {str(e)}",
                reply_markup=self.get_back_button()
            )
    
    async def show_emoji_analysis(self, query, context):
        """Показати аналіз емодзі-реакцій"""
        logger.info("Відображення аналізу емодзі-реакцій")
        
        await query.edit_message_text(
            "⏳ Завантаження аналізу емодзі-реакцій...",
            reply_markup=None
        )
        
        try:
            # Отримання даних про реакції
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT r.reaction, SUM(r.count) as total_count
                FROM reactions r
                GROUP BY r.reaction
                ORDER BY total_count DESC
            """)
            
            emoji_data = cursor.fetchall()
            
            if not emoji_data:
                await query.edit_message_text(
                    "❌ Немає даних про емодзі-реакції. Можливо, у вашому каналі немає постів з реакціями.",
                    reply_markup=self.get_back_button()
                )
                return
            
            # Створення графіку емодзі-реакцій
            import matplotlib.pyplot as plt
            import pandas as pd
            
            df = pd.DataFrame(emoji_data, columns=['emoji', 'count'])
            
            # Обмеження кількості емодзі для відображення
            top_n = min(10, len(df))
            df = df.head(top_n)
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(df)), df['count'], color='skyblue')
            plt.xticks(range(len(df)), df['emoji'], fontsize=14)
            plt.title('Найпопулярніші емодзі-реакції', fontsize=16)
            plt.ylabel('Кількість', fontsize=14)
            
            # Додавання значень над стовпцями
            for i, bar in enumerate(bars):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                        str(int(df['count'].iloc[i])), ha='center', fontsize=12)
            
            plt.tight_layout()
            
            # Збереження графіку у буфер пам'яті
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Отримання даних про реакції в постах
            cursor.execute("""
                SELECT p.message_id, p.text, SUM(r.count) as reaction_count
                FROM posts p
                JOIN reactions r ON p.message_id = r.message_id
                GROUP BY p.message_id
                ORDER BY reaction_count DESC
                LIMIT 5
            """)
            
            top_posts = cursor.fetchall()
            
            # Формування повідомлення
            total_reactions = sum(count for _, count in emoji_data)
            
            stats_text = f"😀 *Аналіз емодзі-реакцій*\n\n"
            stats_text += f"Всього реакцій: {total_reactions}\n\n"
            
            stats_text += "*Найпопулярніші емодзі:*\n"
            for emoji, count in emoji_data[:5]:
                percentage = (count / total_reactions) * 100
                stats_text += f"{emoji}: {count} ({percentage:.1f}%)\n"
            
            if top_posts:
                stats_text += "\n*Пости з найбільшою кількістю реакцій:*\n"
                for i, (msg_id, text, count) in enumerate(top_posts, 1):
                    # Обмеження довжини тексту
                    preview = text[:50] + "..." if text and len(text) > 50 else text
                    stats_text += f"{i}. ID: {msg_id}, Реакцій: {count}\n"
                    if preview:
                        stats_text += f"   Текст: {preview}\n"
            
            # Відправлення графіку
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=buf,
                caption=stats_text,
                parse_mode='Markdown',
                reply_markup=self.get_back_button()
            )
            
            # Редагування оригінального повідомлення
            await query.edit_message_text(
                "✅ Аналіз емодзі-реакцій згенеровано успішно.",
                reply_markup=self.get_back_button()
            )
            
        except Exception as e:
            logger.error(f"Помилка при відображенні аналізу емодзі: {str(e)}")
            logger.error(traceback.format_exc())
            
            await query.edit_message_text(
                f"❌ Помилка при аналізі емодзі: {str(e)}",
                reply_markup=self.get_back_button()
            )

    async def handle_schedule_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробник команди /schedule"""
        logger.info(f"Отримано команду /schedule від користувача {update.effective_user.id}")
    
        # Ініціалізація даних поста
        context.user_data['post_data'] = {
            'text': None,
            'media_path': None,
            'buttons': None,
            'scheduled_time': None
        }
    
        await update.message.reply_text(
            "📝 *Планування нової публікації*\n\n"
            "Введіть текст вашого поста. Можете використовувати markdown-форматування:\n"
            "- *жирний текст* (огорніть текст зірочками)\n"
            "- _курсив_ (огорніть текст підкресленнями)\n"
            "- [посилання](URL) (квадратні дужки для тексту, URL в круглих дужках)\n\n"
            "Введіть /cancel для скасування.",
            parse_mode='Markdown'
        )
    
        # Встановлення стану для обробки наступного повідомлення
        context.user_data['waiting_for'] = 'post_text'
    
    async def show_posts_analysis(self, query, context):
        """Показати аналіз ефективності постів"""
        logger.info("Відображення аналізу ефективності постів")
        
        await query.edit_message_text(
            "⏳ Завантаження аналізу ефективності постів...",
            reply_markup=None
        )
        
        try:
            # Отримання даних про пости
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT p.message_id, p.date, p.views, p.forwards, 
                       SUM(CASE WHEN r.count IS NULL THEN 0 ELSE r.count END) as reactions
                FROM posts p
                LEFT JOIN reactions r ON p.message_id = r.message_id
                GROUP BY p.message_id
                ORDER BY p.date DESC
            """)
            
            posts_data = cursor.fetchall()
            
            if not posts_data:
                await query.edit_message_text(
                    "❌ Немає даних про пости. Переконайтеся, що серверна частина запущена та налаштована правильно.",
                    reply_markup=self.get_back_button()
                )
                return
            
            # Створення DataFrame для аналізу
            import pandas as pd
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import numpy as np
            
            df = pd.DataFrame(posts_data, columns=['message_id', 'date', 'views', 'forwards', 'reactions'])
            df['date'] = pd.to_datetime(df['date'])
            
            # Розрахунок показників ефективності
            df['engagement'] = df['reactions'] + df['forwards']
            df['engagement_rate'] = (df['engagement'] / df['views'] * 100).fillna(0)
            
            # Створення графіків
            fig = plt.figure(figsize=(12, 10))
            
            # Графік 1: Переглядів і залученості в часі
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(df['date'], df['views'], 'b-', marker='o', label='Перегляди')
            ax1.set_ylabel('Перегляди', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            ax2 = ax1.twinx()
            ax2.plot(df['date'], df['engagement_rate'], 'r-', marker='x', label='Залученість')
            ax2.set_ylabel('Залученість (%)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            ax1.set_title('Динаміка переглядів і залученості')
            
            # Налаштування формату дат
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
            if len(df) > 10:
                ax1.xaxis.set_major_locator(mdates.DayLocator(interval=len(df) // 10))
            plt.setp(ax1.get_xticklabels(), rotation=45)
            
            # Графік 2: Розподіл залученості
            ax3 = plt.subplot(2, 1, 2)
            df_sorted = df.sort_values('engagement_rate', ascending=False).head(10)
            
            # Створення підписів для осі X (ID повідомлень)
            x_labels = [f'ID: {mid}' for mid in df_sorted['message_id']]
            
            bars = ax3.bar(range(len(df_sorted)), df_sorted['engagement_rate'], color='skyblue')
            ax3.set_xticks(range(len(df_sorted)))
            ax3.set_xticklabels(x_labels, rotation=45)
            ax3.set_title('Топ-10 постів за залученістю')
            ax3.set_ylabel('Залученість (%)')
            
            # Додавання значень над стовпцями
            for i, bar in enumerate(bars):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f"{df_sorted['engagement_rate'].iloc[i]:.1f}%", ha='center', fontsize=9)
            
            plt.tight_layout()
            
            # Збереження графіку у буфер пам'яті
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Формування статистичної інформації
            avg_views = df['views'].mean()
            avg_engagement = df['engagement'].mean()
            avg_engagement_rate = df['engagement_rate'].mean()
            
            top_post = df.loc[df['engagement_rate'].idxmax()]
            worst_post = df.loc[df['engagement_rate'].idxmin()]
            
            # Кореляція між переглядами і залученістю
            views_engagement_corr = df['views'].corr(df['engagement'])
            
            # Тренд переглядів
            if len(df) >= 5:
                recent_avg = df.head(5)['views'].mean()
                old_avg = df.tail(5)['views'].mean()
                views_trend = recent_avg - old_avg
                views_trend_percent = (views_trend / old_avg) * 100 if old_avg > 0 else 0
                trend_text = f"{'📈' if views_trend >= 0 else '📉'} Тренд переглядів: {views_trend:+.1f} ({views_trend_percent:+.1f}%)"
            else:
                trend_text = "📊 Недостатньо даних для аналізу тренду"
            
            stats_text = (
                f"📊 *Аналіз ефективності постів*\n\n"
                f"📝 Всього проаналізовано постів: {len(df)}\n"
                f"👁️ Середня кількість переглядів: {avg_views:.1f}\n"
                f"🔄 Середня залученість: {avg_engagement:.1f} реакцій\n"
                f"📊 Середній рівень залученості: {avg_engagement_rate:.2f}%\n"
                f"{trend_text}\n\n"
                f"*Найуспішніший пост:*\n"
                f"ID: {int(top_post['message_id'])}\n"
                f"Переглядів: {int(top_post['views'])}\n"
                f"Залученість: {top_post['engagement_rate']:.2f}%\n\n"
                f"*Найменш успішний пост:*\n"
                f"ID: {int(worst_post['message_id'])}\n"
                f"Переглядів: {int(worst_post['views'])}\n"
                f"Залученість: {worst_post['engagement_rate']:.2f}%"
            )
            
            # Відправлення графіку
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=buf,
                caption=stats_text,
                parse_mode='Markdown',
                reply_markup=self.get_back_button()
            )
            
            # Редагування оригінального повідомлення
            await query.edit_message_text(
                "✅ Аналіз ефективності постів згенеровано успішно.",
                reply_markup=self.get_back_button()
            )
            
        except Exception as e:
            logger.error(f"Помилка при відображенні аналізу постів: {str(e)}")
            logger.error(traceback.format_exc())
            
            await query.edit_message_text(
                f"❌ Помилка при аналізі постів: {str(e)}",
                reply_markup=self.get_back_button()
            )
    
    async def show_optimal_time(self, query, context):
        """Показати рекомендації щодо оптимального часу публікації"""
        logger.info("Відображення аналізу оптимального часу публікації")
        
        await query.edit_message_text(
            "⏳ Аналіз оптимального часу публікації...",
            reply_markup=None
        )
        
        try:
            # Отримання даних про пости
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT p.message_id, p.date, p.views, 
                       SUM(CASE WHEN r.count IS NULL THEN 0 ELSE r.count END) as reactions
                FROM posts p
                LEFT JOIN reactions r ON p.message_id = r.message_id
                WHERE p.date IS NOT NULL
                GROUP BY p.message_id
            """)
            
            posts_data = cursor.fetchall()
            
            if not posts_data or len(posts_data) < 10:
                await query.edit_message_text(
                    "❌ Недостатньо даних для аналізу оптимального часу публікації. "
                    "Потрібно щонайменше 10 постів з реакціями.",
                    reply_markup=self.get_back_button()
                )
                return
            
            # Створення DataFrame для аналізу
            import pandas as pd
            import matplotlib.pyplot as plt
            import numpy as np
            
            df = pd.DataFrame(posts_data, columns=['message_id', 'date', 'views', 'reactions'])
            df['date'] = pd.to_datetime(df['date'])
            
            # Додавання часових ознак
            df['hour'] = df['date'].dt.hour
            df['day_of_week'] = df['date'].dt.dayofweek  # 0 = понеділок, 6 = неділя
            
            # Розрахунок показників ефективності
            df['engagement_rate'] = (df['reactions'] / df['views'] * 100).fillna(0)
            
            # Аналіз за годинами
            hour_stats = df.groupby('hour')['engagement_rate'].agg(['mean', 'count']).reset_index()
            hour_stats = hour_stats[hour_stats['count'] >= 2]  # Мінімум 2 пости для години
            
            # Аналіз за днями тижня
            day_stats = df.groupby('day_of_week')['engagement_rate'].agg(['mean', 'count']).reset_index()
            day_stats = day_stats[day_stats['count'] >= 2]  # Мінімум 2 пости для дня
            
            # Перетворення номерів днів у назви
            day_names = {
                0: 'Понеділок',
                1: 'Вівторок',
                2: 'Середа',
                3: 'Четвер',
                4: 'П\'ятниця',
                5: 'Субота',
                6: 'Неділя'
            }
            
            day_stats['day_name'] = day_stats['day_of_week'].map(day_names)
            
            # Створення графіків
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Графік ефективності за годинами
            sorted_hours = hour_stats.sort_values('mean', ascending=False)
            bars1 = ax1.bar(sorted_hours['hour'], sorted_hours['mean'], color='skyblue')
            ax1.set_title('Ефективність публікацій за годинами')
            ax1.set_xlabel('Година доби')
            ax1.set_ylabel('Середня залученість (%)')
            ax1.set_xticks(sorted_hours['hour'])
            
            # Додавання значень над стовпцями
            for i, bar in enumerate(bars1):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f"{sorted_hours['mean'].iloc[i]:.1f}%", ha='center', fontsize=9)
            
            # Графік ефективності за днями тижня
            sorted_days = day_stats.sort_values('mean', ascending=False)
            bars2 = ax2.bar(sorted_days['day_name'], sorted_days['mean'], color='lightgreen')
            ax2.set_title('Ефективність публікацій за днями тижня')
            ax2.set_xlabel('День тижня')
            ax2.set_ylabel('Середня залученість (%)')
            plt.xticks(rotation=45)
            
            # Додавання значень над стовпцями
            for i, bar in enumerate(bars2):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f"{sorted_days['mean'].iloc[i]:.1f}%", ha='center', fontsize=9)
            
            plt.tight_layout()
            
            # Збереження графіку у буфер пам'яті
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Формування рекомендацій
            top_hours = hour_stats.sort_values('mean', ascending=False).head(3)
            top_days = day_stats.sort_values('mean', ascending=False).head(3)
            
            # Комбінований аналіз день-година
            if len(df) >= 20:  # Мінімум 20 постів для комбінованого аналізу
                day_hour_stats = df.groupby(['day_of_week', 'hour'])['engagement_rate'].mean().reset_index()
                top_combinations = day_hour_stats.sort_values('engagement_rate', ascending=False).head(3)
                top_combinations['day_name'] = top_combinations['day_of_week'].map(day_names)
                
                combination_text = "*Найкращі комбінації день-година:*\n"
                for _, row in top_combinations.iterrows():
                    combination_text += f"• {row['day_name']}, {row['hour']}:00 - {row['engagement_rate']:.2f}%\n"
            else:
                combination_text = ""
            
            stats_text = (
                f"⏰ *Рекомендації щодо оптимального часу публікації*\n\n"
                f"Аналіз базується на {len(df)} постах\n\n"
                f"*Найкращі години для публікації:*\n"
            )
            
            for _, row in top_hours.iterrows():
                stats_text += f"• {int(row['hour'])}:00 - {row['mean']:.2f}% залученість\n"
            
            stats_text += f"\n*Найкращі дні тижня для публікації:*\n"
            
            for _, row in top_days.iterrows():
                stats_text += f"• {row['day_name']} - {row['mean']:.2f}% залученість\n"
            
            if combination_text:
                stats_text += f"\n{combination_text}"
            
            stats_text += (
                f"\n💡 *Загальні рекомендації:*\n"
                f"• Публікуйте пости в періоди високої активності аудиторії\n"
                f"• Експериментуйте з різними часовими слотами\n"
                f"• Враховуйте характер контенту при виборі часу публікації"
            )
            
            # Відправлення графіку
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=buf,
                caption=stats_text,
                parse_mode='Markdown',
                reply_markup=self.get_back_button()
            )
            
            # Редагування оригінального повідомлення
            await query.edit_message_text(
                "✅ Аналіз оптимального часу публікації згенеровано успішно.",
                reply_markup=self.get_back_button()
            )
            
        except Exception as e:
            logger.error(f"Помилка при аналізі оптимального часу: {str(e)}")
            logger.error(traceback.format_exc())
    
        await query.edit_message_text(
                    f"❌ Помилка при аналізі оптимального часу: {str(e)}",
                    reply_markup=self.get_back_button()
                )
    
        async def schedule_post_step1(self, query, context):
            """Крок 1: Початок планування поста"""
            logger.info("Початок планування поста")
        
            # Ініціалізація даних поста
            context.user_data['post_data'] = {
                'text': None,
                'media_path': None,
                'buttons': None,
                'scheduled_time': None
            }
        
        await query.edit_message_text(
            "📝 *Планування нової публікації*\n\n"
            "Введіть текст вашого поста. Можете використовувати markdown-форматування:\n"
            "- *жирний текст* (огорніть текст зірочками)\n"
            "- _курсив_ (огорніть текст підкресленнями)\n"
            "- [посилання](URL) (квадратні дужки для тексту, URL в круглих дужках)\n\n"
            "Введіть /cancel для скасування.",
            parse_mode='Markdown',
            reply_markup=None
        )
        
        # Встановлення стану для обробки наступного повідомлення
        context.user_data['waiting_for'] = 'post_text'
    
    async def handle_post_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробник тексту поста"""
        logger.info(f"Отримано текст поста від користувача {update.effective_user.id}")
        
        # Збереження тексту поста
        text = update.message.text
        context.user_data['post_data']['text'] = text
        
        # Питання про додавання медіа
        keyboard = [
            [InlineKeyboardButton("Так, додати медіа", callback_data="add_media")],
            [InlineKeyboardButton("Ні, тільки текст", callback_data="no_media")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "✅ Текст поста збережено.\n\n"
            "Чи бажаєте додати медіа-файл до поста?",
            reply_markup=reply_markup
        )
    
    async def handle_media(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробник медіа-файлу для поста"""
        logger.info(f"Отримано медіа-файл від користувача {update.effective_user.id}")
        
        # Отримання файлу
        if update.message.photo:
            file_id = update.message.photo[-1].file_id
            file_extension = 'jpg'
        elif update.message.video:
            file_id = update.message.video.file_id
            file_extension = 'mp4'
        elif update.message.document:
            file_id = update.message.document.file_id
            file_extension = update.message.document.file_name.split('.')[-1]
        else:
            await update.message.reply_text(
                "❌ Непідтримуваний тип файлу. Будь ласка, надішліть фото, відео або документ."
            )
            return
        
        # Завантаження файлу
        file = await context.bot.get_file(file_id)
        file_path = f"media/{update.effective_user.id}_{int(time.time())}.{file_extension}"
        
        # Створення директорії, якщо вона не існує
        os.makedirs("media", exist_ok=True)
        
        await file.download_to_drive(file_path)
        
        # Збереження шляху до файлу
        context.user_data['post_data']['media_path'] = file_path
        
        # Перехід до вибору часу публікації
        await self.schedule_post_step2(update, context)
    
    async def schedule_post_step2(self, update, context):
        """Крок 2: Вибір дня тижня для публікації"""
        logger.info("Планування поста: вибір дня тижня")
        
        # Створення меню для вибору дня тижня
        today = datetime.datetime.now()
        keyboard = []
        
        for i in range(7):
            date = today + datetime.timedelta(days=i)
            day_name = date.strftime("%A")
            day_name_ua = {
                "Monday": "Понеділок",
                "Tuesday": "Вівторок",
                "Wednesday": "Середа",
                "Thursday": "Четвер",
                "Friday": "П'ятниця",
                "Saturday": "Субота",
                "Sunday": "Неділя"
            }.get(day_name, day_name)
            
            date_str = date.strftime("%d.%m.%Y")
            button_text = f"{day_name_ua} ({date_str})"
            
            keyboard.append([InlineKeyboardButton(button_text, callback_data=f"day_{i}")])
        
        keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Визначення типу оновлення (повідомлення або callback)
        if hasattr(update, 'message'):
            await update.message.reply_text(
                "📅 Виберіть день для публікації:",
                reply_markup=reply_markup
            )
        else:
            await update.edit_message_text(
                "📅 Виберіть день для публікації:",
                reply_markup=reply_markup
            )
    
    async def schedule_post_step3(self, query, context):
        """Крок 3: Вибір години для публікації"""
        logger.info("Планування поста: вибір години")
        
        # Отримання вибраного дня
        day_offset = int(query.data.split('_')[1])
        context.user_data['post_data']['day_offset'] = day_offset
        
        # Створення меню для вибору години
        keyboard = []
        row = []
        
        for hour in range(24):
            hour_str = f"{hour:02d}:00"
            row.append(InlineKeyboardButton(hour_str, callback_data=f"hour_{hour}"))
            
            if len(row) == 6:  # 6 кнопок у рядку
                keyboard.append(row)
                row = []
        
        if row:  # Додавання останнього неповного рядка
            keyboard.append(row)
        
        keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="schedule")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Розрахунок дати публікації
        today = datetime.datetime.now()
        selected_date = today + datetime.timedelta(days=day_offset)
        day_name = selected_date.strftime("%A")
        day_name_ua = {
            "Monday": "Понеділок",
            "Tuesday": "Вівторок",
            "Wednesday": "Середа",
            "Thursday": "Четвер",
            "Friday": "П'ятниця",
            "Saturday": "Субота",
            "Sunday": "Неділя"
        }.get(day_name, day_name)
        date_str = selected_date.strftime("%d.%m.%Y")
        
        await query.edit_message_text(
            f"🕒 Виберіть годину для публікації:\n"
            f"Обраний день: {day_name_ua} ({date_str})",
            reply_markup=reply_markup
        )
    
    async def schedule_post_step4(self, query, context):
        """Крок 4: Вибір хвилин для публікації"""
        logger.info("Планування поста: вибір хвилин")
        
        # Отримання вибраної години
        hour = int(query.data.split('_')[1])
        context.user_data['post_data']['hour'] = hour
        
        # Створення меню для вибору хвилин
        keyboard = []
        row = []
        
        for minute in [0, 15, 30, 45]:
            minute_str = f"{minute:02d}"
            row.append(InlineKeyboardButton(minute_str, callback_data=f"minute_{minute}"))
        
        keyboard.append(row)
        keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data=f"day_{context.user_data['post_data']['day_offset']}")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Розрахунок дати публікації
        today = datetime.datetime.now()
        selected_date = today + datetime.timedelta(days=context.user_data['post_data']['day_offset'])
        day_name = selected_date.strftime("%A")
        day_name_ua = {
            "Monday": "Понеділок",
            "Tuesday": "Вівторок",
            "Wednesday": "Середа",
            "Thursday": "Четвер",
            "Friday": "П'ятниця",
            "Saturday": "Субота",
            "Sunday": "Неділя"
        }.get(day_name, day_name)
        date_str = selected_date.strftime("%d.%m.%Y")
        hour_str = f"{hour:02d}"
        
        await query.edit_message_text(
            f"🕒 Виберіть хвилини для публікації:\n"
            f"Обраний день: {day_name_ua} ({date_str})\n"
            f"Обрана година: {hour_str}:00",
            reply_markup=reply_markup
        )
    
    async def schedule_post_confirm(self, query, context):
        """Підтвердження запланованої публікації"""
        logger.info("Планування поста: підтвердження")
        
        # Отримання вибраних хвилин
        minute = int(query.data.split('_')[1])
        context.user_data['post_data']['minute'] = minute
        
        # Розрахунок повної дати публікації
        today = datetime.datetime.now()
        day_offset = context.user_data['post_data']['day_offset']
        hour = context.user_data['post_data']['hour']
        
        selected_date = today + datetime.timedelta(days=day_offset)
        scheduled_time = datetime.datetime(
            year=selected_date.year,
            month=selected_date.month,
            day=selected_date.day,
            hour=hour,
            minute=minute
        )
        
        # Перевірка, чи час у майбутньому
        if scheduled_time <= datetime.datetime.now():
            await query.edit_message_text(
                "❌ Вибраний час вже минув. Будь ласка, виберіть час у майбутньому.",
                reply_markup=self.get_back_button()
            )
            return
        
        # Форматування часу
        time_str = scheduled_time.strftime("%d.%m.%Y %H:%M")
        
        # Збереження часу публікації
        context.user_data['post_data']['scheduled_time'] = scheduled_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Створення підтверджувального повідомлення
        post_text = context.user_data['post_data']['text']
        if len(post_text) > 200:
            post_preview = post_text[:200] + "..."
        else:
            post_preview = post_text
        
        has_media = context.user_data['post_data']['media_path'] is not None
        media_text = "Так" if has_media else "Ні"
        
        confirmation_text = (
            f"📝 *Підтвердження запланованої публікації*\n\n"
            f"*Час публікації:* {time_str}\n"
            f"*Медіа:* {media_text}\n\n"
            f"*Текст поста:*\n{post_preview}\n\n"
            f"Підтвердіть публікацію:"
        )
        
        # Кнопки підтвердження
        keyboard = [
            [InlineKeyboardButton("✅ Підтвердити", callback_data="confirm_post")],
            [InlineKeyboardButton("❌ Скасувати", callback_data="cancel_post")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            confirmation_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def confirm_post(self, query, context):
        """Підтвердження і збереження запланованого поста"""
        logger.info("Планування поста: фінальне підтвердження")
        
        try:
            # Отримання даних поста
            post_data = context.user_data['post_data']
            
            # Підключення до бази даних
            cursor = self.db_conn.cursor()
            
            # Збереження запланованого поста
            cursor.execute("""
                INSERT INTO scheduled_posts (scheduled_time, post_text, media_path, status)
                VALUES (?, ?, ?, 'pending')
            """, (
                post_data['scheduled_time'],
                post_data['text'],
                post_data['media_path']
            ))
            
            self.db_conn.commit()
            post_id = cursor.lastrowid
            
            # Форматування часу для відображення
            scheduled_time = datetime.datetime.strptime(post_data['scheduled_time'], "%Y-%m-%d %H:%M:%S")
            time_str = scheduled_time.strftime("%d.%m.%Y %H:%M")
            
            # Повідомлення про успіх
            await query.edit_message_text(
                f"✅ Пост успішно запланований!\n\n"
                f"ID: {post_id}\n"
                f"Час публікації: {time_str}\n\n"
                f"Серверна частина опублікує пост автоматично у вказаний час.",
                reply_markup=self.get_back_button()
            )
            
            # Очищення даних
            if 'post_data' in context.user_data:
                del context.user_data['post_data']
            
            logger.info(f"Пост успішно запланований на {time_str}, ID: {post_id}")
            
        except Exception as e:
            logger.error(f"Помилка при плануванні поста: {str(e)}")
            logger.error(traceback.format_exc())
            
            await query.edit_message_text(
                f"❌ Помилка при плануванні поста: {str(e)}",
                reply_markup=self.get_back_button()
            )
    
    async def cancel_post(self, query, context):
        """Скасування запланованого поста"""
        logger.info("Планування поста: скасування")
        
        # Очищення даних
        if 'post_data' in context.user_data:
            # Видалення завантаженого медіа-файлу, якщо він існує
            if context.user_data['post_data'].get('media_path') and os.path.exists(context.user_data['post_data']['media_path']):
                try:
                    os.remove(context.user_data['post_data']['media_path'])
                    logger.info(f"Видалено медіа-файл: {context.user_data['post_data']['media_path']}")
                except Exception as e:
                    logger.error(f"Помилка при видаленні медіа-файлу: {str(e)}")
            
            del context.user_data['post_data']
        
        await query.edit_message_text(
            "❌ Планування поста скасовано.",
            reply_markup=self.get_back_button()
        )
    
    async def search_channels_step1(self, query, context):
        """Крок 1: Початок пошуку релевантних каналів"""
        logger.info("Початок пошуку релевантних каналів")
        
        await query.edit_message_text(
            "🔍 *Пошук релевантних каналів*\n\n"
            "Введіть ключові слова для пошуку через кому.\n"
            "Наприклад: новини, спорт, технології\n\n"
            "Введіть /cancel для скасування.",
            parse_mode='Markdown',
            reply_markup=None
        )
        
        # Встановлення стану для обробки наступного повідомлення
        context.user_data['waiting_for'] = 'search_keywords'
    
    async def handle_search_keywords(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробник ключових слів для пошуку каналів"""
        logger.info(f"Отримано ключові слова для пошуку від користувача {update.effective_user.id}")
        
        keywords = update.message.text.split(',')
        keywords = [k.strip() for k in keywords if k.strip()]
        
        if not keywords:
            await update.message.reply_text(
                "❌ Не вказано жодного ключового слова. Будь ласка, спробуйте знову.",
                reply_markup=self.get_back_button()
            )
            return
        
        await update.message.reply_text(
            f"⏳ Пошук релевантних каналів за ключовими словами: {', '.join(keywords)}...\n\n"
            "Це може зайняти деякий час. Будь ласка, зачекайте."
        )
        
        try:
            # Пошук каналів у базі даних
            cursor = self.db_conn.cursor()
            
            # Підготовка умови для SQL-запиту
            placeholders = ', '.join(['?'] * len(keywords))
            like_conditions = ' OR '.join([f"channel_title LIKE ?" for _ in keywords])
            
            # Параметри для запиту: шаблони для LIKE
            params = [f"%{keyword}%" for keyword in keywords]
            
            cursor.execute(f"""
                SELECT channel_id, channel_title, subscribers, relevance_score
                FROM relevant_channels
                WHERE {like_conditions}
                ORDER BY relevance_score DESC, subscribers DESC
                LIMIT 20
            """, params)
            
            channels = cursor.fetchall()
            
            if not channels:
                await update.message.reply_text(
                    "🔍 На жаль, релевантних каналів не знайдено. "
                    "Спробуйте змінити ключові слова або запустіть "
                    "пошук релевантних каналів з серверної частини.",
                    reply_markup=self.get_back_button()
                )
                return
            
            # Формування списку знайдених каналів
            results_text = f"🔍 *Знайдено {len(channels)} релевантних каналів:*\n\n"
            
            for i, (channel_id, title, subscribers, score) in enumerate(channels, 1):
                subscribers_str = f"{subscribers:,}" if subscribers else "Невідомо"
                results_text += f"{i}. *{title}*\n"
                results_text += f"   ID: {channel_id}\n"
                results_text += f"   Підписників: {subscribers_str}\n"
                results_text += f"   Релевантність: {score:.2f}\n\n"
            
            # Відправлення результатів
            keyboard = [
                [InlineKeyboardButton("🔄 Новий пошук", callback_data="search_channels")],
                [InlineKeyboardButton("⬅️ Повернутися до головного меню", callback_data="back_to_main")]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                results_text,
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"Помилка при пошуку каналів: {str(e)}")
            logger.error(traceback.format_exc())
            
            await update.message.reply_text(
                f"❌ Помилка при пошуку каналів: {str(e)}",
                reply_markup=self.get_back_button()
            )
    
    async def back_to_main_menu(self, query, context):
        """Повернення до головного меню"""
        logger.info("Повернення до головного меню")
        
        # Створення головного меню
        keyboard = [
            [InlineKeyboardButton("📊 Статистика каналу", callback_data="stats")],
            [InlineKeyboardButton("📝 Запланувати пост", callback_data="schedule")],
            [InlineKeyboardButton("🔍 Аналіз аудиторії", callback_data="audience")],
            [InlineKeyboardButton("😀 Аналіз емодзі", callback_data="emoji")],
            [InlineKeyboardButton("📈 Аналіз постів", callback_data="posts")],
            [InlineKeyboardButton("⏰ Оптимальний час", callback_data="optimal_time")],
            [InlineKeyboardButton("🌐 Пошук релевантних каналів", callback_data="search_channels")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            f"👋 Вітаю! Я {BOT_NAME} - бот для автоматичного наповнення та просування вашого Telegram-каналу.\n\n"
            "Оберіть дію з меню нижче:",
            reply_markup=reply_markup
        )
    
    def get_back_button(self):
        """Створити кнопку для повернення до головного меню"""
        keyboard = [[InlineKeyboardButton("⬅️ Повернутися до головного меню", callback_data="back_to_main")]]
        return InlineKeyboardMarkup(keyboard)
    
    async def error_handler(self, update, context):
        """Обробник помилок"""
        logger.error(f"Помилка при обробці оновлення: {context.error}")
        logger.error(traceback.format_exc())
        
        try:
            # Надсилання повідомлення про помилку користувачу
            if update and update.effective_chat:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=f"❌ Виникла помилка: {context.error}",
                    reply_markup=self.get_back_button()
                )
        except Exception as e:
            logger.error(f"Помилка при відправці повідомлення про помилку: {str(e)}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробник звичайних повідомлень"""
        logger.info(f"Отримано повідомлення від користувача {update.effective_user.id}: {update.message.text[:50]}...")
        
        # Перевірка, чи очікуємо на конкретний тип повідомлення
        waiting_for = context.user_data.get('waiting_for')
        
        if waiting_for == 'post_text':
            await self.handle_post_text(update, context)
        elif waiting_for == 'post_media':
            await self.handle_media(update, context)
        elif waiting_for == 'search_keywords':
            await self.handle_search_keywords(update, context)
        else:
            # Якщо немає очікувань, відправляємо довідку
            await self.help_command(update, context)
    
    async def handle_media_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробник медіа-повідомлень"""
        logger.info(f"Отримано медіа від користувача {update.effective_user.id}")
        
        # Перевірка, чи очікуємо на медіа
        waiting_for = context.user_data.get('waiting_for')
        
        if waiting_for == 'post_media':
            await self.handle_media(update, context)
        else:
            await update.message.reply_text(
                "🤔 Я отримав медіа-файл, але не знаю, що з ним робити. "
                "Використовуйте команду /schedule, щоб запланувати пост з медіа.",
                reply_markup=self.get_back_button()
            )
    
    async def handle_command_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробник команди /cancel"""
        logger.info(f"Отримано команду /cancel від користувача {update.effective_user.id}")
        
        # Очищення стану
        if 'waiting_for' in context.user_data:
            del context.user_data['waiting_for']
        
        # Очищення даних поста, якщо є
        if 'post_data' in context.user_data:
            # Видалення завантаженого медіа-файлу, якщо він існує
            if context.user_data['post_data'].get('media_path') and os.path.exists(context.user_data['post_data']['media_path']):
                try:
                    os.remove(context.user_data['post_data']['media_path'])
                    logger.info(f"Видалено медіа-файл: {context.user_data['post_data']['media_path']}")
                except Exception as e:
                    logger.error(f"Помилка при видаленні медіа-файлу: {str(e)}")
            
            del context.user_data['post_data']
        
        # Створення кнопки для повернення
        keyboard = [[InlineKeyboardButton("Повернутися до головного меню", callback_data="back_to_main")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "❌ Операцію скасовано. Всі тимчасові дані очищено.",
            reply_markup=reply_markup
        )
    
    async def handle_command_skip(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробник команди /skip для пропуску додавання медіа"""
        logger.info(f"Отримано команду /skip від користувача {update.effective_user.id}")
        
        waiting_for = context.user_data.get('waiting_for')
        
        if waiting_for == 'post_media':
            # Пропуск додавання медіа і перехід до вибору часу
            await self.schedule_post_step2(update, context)
        else:
            await update.message.reply_text(
                "❓ Команда /skip доступна лише під час додавання медіа-файлу до поста.",
                reply_markup=self.get_back_button()
            )
    
    def run(self):
        """Запуск клієнтської частини бота"""
        logger.info("Запуск клієнтської частини бота")
        
        try:
            # Перевірка наявності необхідних бібліотек
            required_libraries = ['telegram', 'pandas', 'matplotlib']
            missing_libraries = []
            
            for lib in required_libraries:
                try:
                    __import__(lib)
                except ImportError:
                    missing_libraries.append(lib)
            
            if missing_libraries:
                logger.error(f"Відсутні необхідні бібліотеки: {', '.join(missing_libraries)}")
                logger.error("Встановіть їх за допомогою pip:")
                logger.error(f"pip install {' '.join(missing_libraries)}")
                sys.exit(1)
            
            # Відображення інформації про конфігурацію
            logger.info("========================")
            logger.info("Запуск бота з наступними параметрами:")
            logger.info(f"BOT_TOKEN: {self.token[:8]}...{self.token[-8:]}")  # Показуємо частково для безпеки
            logger.info(f"BOT_NAME: {BOT_NAME}")
            logger.info("========================")
            
            # Створення екземпляру застосунку
            self.application = Application.builder().token(self.token).build()
            
            # Додавання обробників команд
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CommandHandler("stats", lambda update, context: self.show_channel_stats(update.callback_query, context)))
            self.application.add_handler(CommandHandler("schedule", self.handle_schedule_command))
            self.application.add_handler(CommandHandler("audience", lambda update, context: self.show_audience_analysis(update.callback_query, context)))
            self.application.add_handler(CommandHandler("emoji", lambda update, context: self.show_emoji_analysis(update.callback_query, context)))
            self.application.add_handler(CommandHandler("posts", lambda update, context: self.show_posts_analysis(update.callback_query, context)))
            self.application.add_handler(CommandHandler("optimal_time", lambda update, context: self.show_optimal_time(update.callback_query, context)))
            self.application.add_handler(CommandHandler("search", lambda update, context: self.search_channels_step1(update.callback_query, context)))
            self.application.add_handler(CommandHandler("cancel", self.handle_command_cancel))
            self.application.add_handler(CommandHandler("skip", self.handle_command_skip))
            
            # Обробник натискань на кнопки
            self.application.add_handler(CallbackQueryHandler(self.button_callback))
            
            # Обробник текстових повідомлень
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            
            # Обробник медіа-повідомлень
            self.application.add_handler(MessageHandler(
                filters.PHOTO | filters.VIDEO | filters.Document.ALL, 
                self.handle_media_message
            ))
            
            # Обробник помилок
            self.application.add_error_handler(self.error_handler)
            
            # Запуск бота
            logger.info("Запуск бота у режимі polling")
            self.application.run_polling(allowed_updates=Update.ALL_TYPES)
            
        except Exception as e:
            logger.critical(f"Критична помилка при запуску бота: {str(e)}")
            logger.critical(traceback.format_exc())


def main():
    """Запуск клієнтської частини"""
    logger.info("Запуск клієнтської частини чат-бота")
    
    # Використовуємо конфігураційну змінну
    token = BOT_TOKEN
    
    if not token:
        logger.error("Не вказано токен бота. Перевірте конфігурацію.")
        sys.exit(1)
    
    # Ініціалізація та запуск клієнта
    client = TelegramClient(token)
    client.run()


if __name__ == "__main__":
    main()