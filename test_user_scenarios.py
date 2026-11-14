import requests
import json

base_url = 'http://127.0.0.1:8080'

scenarios = [
    {
        'id': 'TC-01',
        'title': 'الوضع الاسمي المستقر (Nomex Space Suit, Low Flow, 21% O2)',
        'description': 'فحص بدلة رائد الفضاء في ظروف الكابينة العادية مع تدفق هواء هادئ.',
        'payload': {
            'material_type': 'Nomex', 'geometry': 'Fabric', 'color': 'standard',
            'thickness_mm': 1.2, 'width_cm': 2.0, 'fan_display': 15.0, 'o2_initial': 21.0,
            'liquid_fuel': 'Methanol', 'd0_mm': 2.0, 'x_co2': 0.10, 'x_he': 0.05,
            'gas_fuel': 'ethane', 'pressure_bar': 1.0, 'xf': 0.20, 'mhc_mg_s': 0.5
        }
    },
    {
        'id': 'TC-02',
        'title': 'خطر المواد الصلبة الحرج (PMMA Acrylic Glass, High Fan 70, High O2 30%)',
        'description': 'محاكاة اشتعال لوح أكريليك شفاف في ممر تهوية سريع مع نسبة أكسجين مرتفعة.',
        'payload': {
            'material_type': 'PMMA', 'geometry': 'Film', 'color': 'standard',
            'thickness_mm': 0.5, 'width_cm': 2.0, 'fan_display': 70.0, 'o2_initial': 30.0,
            'liquid_fuel': 'Methanol', 'd0_mm': 1.5, 'x_co2': 0.10, 'x_he': 0.0,
            'gas_fuel': 'ethane', 'pressure_bar': 1.0, 'xf': 0.20, 'mhc_mg_s': 0.5
        }
    },
    {
        'id': 'TC-03',
        'title': 'تسرب وقود سائل متطاير (Heptane Droplet d0=4.5mm, 0% CO2)',
        'description': 'تسرب وقود تبريد هيدروكربوني وتطايره كقطرات معلقة بدون أي غاز خامل.',
        'payload': {
            'material_type': 'Nomex', 'geometry': 'Fabric', 'color': 'standard',
            'thickness_mm': 1.0, 'width_cm': 2.0, 'fan_display': 20.0, 'o2_initial': 21.0,
            'liquid_fuel': 'Heptane', 'd0_mm': 4.5, 'x_co2': 0.0, 'x_he': 0.0,
            'gas_fuel': 'ethane', 'pressure_bar': 1.0, 'xf': 0.25, 'mhc_mg_s': 0.6
        }
    },
    {
        'id': 'TC-04',
        'title': 'بيئة قاعدة القمر عالية الأكسجين (Lunar Base: 34% O2, 0.6 bar)',
        'description': 'محاكاة مقصورة رواد الفضاء في بيئة أرتميس القمرية (أكسجين 34% وضغط منخفض 0.6 بار).',
        'payload': {
            'material_type': 'PMMA', 'geometry': 'Slab', 'color': 'standard',
            'thickness_mm': 2.0, 'width_cm': 2.0, 'fan_display': 45.0, 'o2_initial': 34.0,
            'liquid_fuel': 'Heptane', 'd0_mm': 3.0, 'x_co2': 0.05, 'x_he': 0.0,
            'gas_fuel': 'propane', 'pressure_bar': 0.6, 'xf': 0.40, 'mhc_mg_s': 1.2
        }
    },
    {
        'id': 'TC-05',
        'title': 'حريق مجمع ثلاثي الأطوار (Compound Tri-Phase Runaway Fire)',
        'description': 'كارثة مركبة: تماس كهربائي في مواد بلاستيكية + تسرب وقود سائل + تسرب غاز مضغوط.',
        'payload': {
            'material_type': 'PMMA', 'geometry': 'Film', 'color': 'standard',
            'thickness_mm': 1.0, 'width_cm': 2.0, 'fan_display': 60.0, 'o2_initial': 28.0,
            'liquid_fuel': 'Heptane', 'd0_mm': 4.0, 'x_co2': 0.0, 'x_he': 0.0,
            'gas_fuel': 'propane', 'pressure_bar': 2.2, 'xf': 0.70, 'mhc_mg_s': 2.5
        }
    }
]

print("================================================================================")
print("       NASA MICROGRAVITY AI CHAMBER // FULL REAL-WORLD USER TEST RUN")
print("================================================================================")

for sc in scenarios:
    res = requests.post(f"{base_url}/api/tri-hazard/evaluate", json=sc['payload'])
    d = res.json()
    print("-" * 80)
    print(f"[{sc['id']}] {sc['title']}")
    print(f"الوصف: {sc['description']}")
    print(f"► مؤشر الخطر الكلي (Composite Hazard Score): {d['composite_score']} / 100")
    print(f"► تصنيف مستوى التهديد: {d['threat_level']}")
    print(f"► تفصيل الأطوار الثلاثة:")
    print(f"   • الطور الصلب (Solid BASS-II): {d['breakdown']['solid_risk']}% خطر | احتمال الانطفاء الذاتي: {d['breakdown']['solid_extinction_prob']}%")
    print(f"   • طور القطرات (Liquid FLEX):  {d['breakdown']['liquid_risk']}% خطر | احتمال الانفجار/التناثر: {d['breakdown']['liquid_disruption_prob']}% | احتمال الانطفاء: {d['breakdown']['liquid_extinction_prob']}%")
    print(f"   • الطور الغازي (Gas CFI-G):   {d['breakdown']['gas_risk']}% خطر | احتمال الانطفاء الذاتي: {d['breakdown']['gas_extinction_prob']}%")
    print(f"► عنق الزجاجة الحرج (Bottleneck Phase): {d['bottleneck']['phase']} بنسبة خطر {d['bottleneck']['risk_percentage']}%")
    print(f"► توجيهات AstroCommander AI التكتيكية:")
    for act in d['commander_directive']['tactical_actions']:
        print(f"   [ACTION] {act}")
    print(f"► الخطر المتوقع بعد التنفيذ التلقائي: {d['commander_directive']['projected_composite_score']} (نسبة التخفيض: -{d['commander_directive']['projected_reduction']} نقطة)")

print("================================================================================")
print("TEST EXECUTION COMPLETED SUCCESSFULLY.")
