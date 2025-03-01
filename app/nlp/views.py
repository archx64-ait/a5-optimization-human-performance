from django.views.generic import TemplateView
from django.views.generic.edit import FormView
from nlp.forms import NLPForm
from typing import Any
from nlp.utils_scratch import *
import torch


from transformers import AutoModelForCausalLM, AutoTokenizer


class IndexView(TemplateView):
    template_name = "index.html"


# class SuccessView(TemplateView):
#     template_name = "success.html"

#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs)

#         result = self.request.GET.get("result")

#         try:
#             # Add the result to the context
#             context["result"] = result

#         except ValueError:
#             context["result"] = [""]

#         return context


class NLPFormView(FormView):

    form_class = NLPForm
    template_name = "nlp.html"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "archx64/best-dpo-Qwen-Qwen2-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def run_inference(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=50)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def form_valid(self, form):
        prompt = form.cleaned_data["prompt"]
        result = self.run_inference(prompt=prompt)
        context = self.get_context_data(result=result)
        print(context)
        return self.render_to_response(context)

    def form_invalid(self, form):
        return super().form_invalid(form)

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        # context["results"] = getattr(self, "result", None)
        context["result"] = kwargs.get("result", None)
        return context
