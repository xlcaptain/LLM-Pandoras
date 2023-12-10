## 📦 Django 全局字段检查装饰器

```angular2html
def require_params(required_params):
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            missing_params = []
            wrong_type_params = []
            for p, expected_type in required_params.items():
                value = kwargs.get(p) or request.request.query_params.get(p) or request.request.data.get(p)
                if value is None:
                    missing_params.append(p)
                elif not isinstance(value, expected_type):
                    wrong_type_params.append(p)
            if missing_params:
                return JsonResponse({'code': 400, 'msg': f'参数错误，缺少参数: {", ".join(missing_params)}'}, status=200)
            if wrong_type_params:
                return JsonResponse({'code': 400, 'msg': f'参数错误，类型错误: {", ".join(wrong_type_params)}'}, status=200)
            return view_func(request, *args, **kwargs)

        return _wrapped_view

    return decorator
```