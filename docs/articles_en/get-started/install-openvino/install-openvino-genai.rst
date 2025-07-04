Install OpenVINO™ GenAI
====================================

OpenVINO GenAI is a tool, simplifying generative AI model inference. It is based on the
OpenVINO Runtime, hiding the complexity of the generation process and minimizing the amount of
code required. You provide a model and the input context directly to the tool, while it
performs tokenization of the input text, executes the generation loop on the selected device,
and returns the generated content. For a quickstart guide, refer to the
:doc:`GenAI API Guide <../../openvino-workflow-generative/inference-with-genai>`.

To see OpenVINO GenAI in action, check these Jupyter notebooks:
`LLM-powered Chatbot <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-chatbot>`__
and
`LLM Instruction-following pipeline <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-question-answering>`__.

OpenVINO GenAI is available for installation via PyPI and Archive distributions.
A `detailed guide <https://github.com/openvinotoolkit/openvino.genai/blob/releases/2025/0/src/docs/BUILD.md>`__
on how to build OpenVINO GenAI is available in the OpenVINO GenAI repository.

PyPI Installation
###############################

To install the GenAI package via PyPI, follow the standard :doc:`installation steps <install-openvino-pip>`,
but use the *openvino-genai* package instead of *openvino*:

.. code-block:: python

   python -m pip install openvino-genai

Archive Installation
###############################

The OpenVINO GenAI archive package includes the OpenVINO™ Runtime, as well as :doc:`Tokenizers <../../openvino-workflow-generative/ov-tokenizers>`.
It installs the same way as the standard OpenVINO Runtime, so follow its installation steps,
just use the OpenVINO GenAI package instead:

Linux
++++++++++++++++++++++++++

.. tab-set::

   .. tab-item:: x86_64
      :sync: x86-64

      .. tab-set::

         .. tab-item:: Ubuntu 24.04
            :sync: ubuntu-24

            .. code-block:: sh

               curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2025.2/linux/openvino_genai_ubuntu24_2025.2.0.0_x86_64.tar.gz --output openvino_genai_2025.2.0.0.tgz
               tar -xf openvino_genai_2025.2.0.0.tgz

         .. tab-item:: Ubuntu 22.04
            :sync: ubuntu-22

            .. code-block:: sh

               curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2025.2/linux/openvino_genai_ubuntu22_2025.2.0.0_x86_64.tar.gz --output openvino_genai_2025.2.0.0.tgz
               tar -xf openvino_genai_2025.2.0.0.tgz

         .. tab-item:: Ubuntu 20.04
            :sync: ubuntu-20

            .. code-block:: sh

               curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2025.2/linux/openvino_genai_ubuntu20_2025.2.0.0_x86_64.tar.gz  --output openvino_genai_2025.2.0.0.tgz
               tar -xf openvino_genai_2025.2.0.0.tgz


   .. tab-item:: ARM 64-bit
      :sync: arm-64

      .. code-block:: sh

         curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2025.2/linux/openvino_genai_ubuntu20_2025.2.0.0_arm64.tar.gz -O openvino_genai_2025.2.0.0.tgz
         tar -xf openvino_genai_2025.2.0.0.tgz


Windows
++++++++++++++++++++++++++

.. code-block:: sh

   cd <user_home>/Downloads
   curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2025.2/windows/openvino_genai_windows_2025.2.0.0_x86_64.zip --output openvino_genai_2025.2.0.0.zip

macOS
++++++++++++++++++++++++++

.. tab-set::

   .. tab-item:: x86, 64-bit
      :sync: x86-64

      .. code-block:: sh

         curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2025.2/macos/openvino_genai_macos_12_6_2025.2.0.0_x86_64.tar.gz --output openvino_genai_2025.2.0.0.tgz
         tar -xf openvino_genai_2025.2.0.0.tgz

   .. tab-item:: ARM, 64-bit
      :sync: arm-64

      .. code-block:: sh

         curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2025.2/macos/openvino_genai_macos_12_6_2025.2.0.0_arm64.tar.gz --output openvino_genai_2025.2.0.0.tgz
         tar -xf openvino_genai_2025.2.0.0.tgz


Here are the full guides:
:doc:`Linux <install-openvino-archive-linux>`,
:doc:`Windows <install-openvino-archive-windows>`, and
:doc:`macOS <install-openvino-archive-macos>`.



