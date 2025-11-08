# Quantization-as-a-Service
OpenFaaS Implementation for QuantifAI

This guide provides detailed instructions for setting up and deploying the QuantifAI-FAAS function using OpenFaaS.

---

## OpenFaaS Setup

Follow these steps to install and configure OpenFaaS:

1. **Install OpenFaaS with Basic Authentication**:
   ```bash
   sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml arkade install openfaas --basic-auth
   ```

2. **Install the OpenFaaS CLI (faas-cli)**:
   ```bash
   sudo arkade get faas-cli
   ```

3. **Retrieve the Basic Authentication Password**:
   ```bash
   sudo kubectl -n openfaas get secret basic-auth -o jsonpath="{.data.basic-auth-password}" | base64 --decode; echo
   ```

4. **Log in to OpenFaaS**:
   Replace `<password>` with the password retrieved in the previous step:
   ```bash
   sudo faas-cli login --password <password>
   ```

---

## Function Creation and Deployment

1. **Build the Function Image**:
   ```bash
   sudo faas-cli build -f ./quantifai-faas.yml --build-arg PYTHON_VERSION=3.11
   ```

2. **Push the Function Image**:
   ```bash
   sudo faas-cli push -f ./quantifai-faas.yml
   ```

3. **Deploy the Function**:
   ```bash
   sudo faas-cli deploy -f ./quantifai-faas.yml
   ```

4. **Verify the Deployed Functions**:
   ```bash
   sudo faas-cli list
   ```

---

## Gateway Configuration

1. **Update Gateway Timeouts**:
   Edit the gateway deployment to increase timeouts to 15 minutes:
   ```bash
   kubectl edit deployment gateway -n openfaas
   ```

   Add or update the following environment variables:
   ```yaml
   env:
   - name: read_timeout
     value: 15m
   - name: write_timeout
     value: 15m
   - name: upstream_timeout
     value: 15m
   ```

2. **Restart the Gateway Deployment**:
   ```bash
   kubectl rollout restart deployment gateway -n openfaas
   ```

3. **Expose the Gateway Locally**:
   ```bash
   sudo kubectl port-forward -n openfaas svc/gateway 8080:8080 &
   ```

---

## Access Gateway from the Local Network

1. **Expose the Gateway via NodePort**:
   ```bash
   sudo kubectl patch svc gateway -n openfaas -p '{"spec": {"type": "NodePort"}}'
   ```

2. **Retrieve the Gateway Port**:
   ```bash
   sudo kubectl get svc -n openfaas gateway
   ```
   Note the `NodePort` value to access the gateway externally.

---

## Testing the Function Locally

1. **Invoke the Function with a Test JSON File**:
   Replace `test.json` with the path to your input JSON file:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d @test.json http://127.0.0.1:8080/function/quantifai-faas
   ```

---

## Notes and Best Practices

- Ensure the **Kubernetes cluster** is properly configured before deploying OpenFaaS.
- For production deployments, consider securing the gateway and limiting external access.
- Use the `faas-cli` for managing functions efficiently.
- Test your function locally before deploying to ensure smooth operation in the OpenFaaS environment.

---

## Cite this Work

```latex
@INPROCEEDINGS{DeNo2507:Serverless,
AUTHOR="Danny {De Novi} and Pierluigi Dell'Acqua and Lorenzo Carnevale and Maria
Fazio and Massimo Villari",
TITLE="A Serverless {Quantization-as-a-Service} Model to Run Compression Jobs for
Edge Intelligence",
BOOKTITLE="2025 IEEE Symposium on Computers and Communications (ISCC) (IEEE ISCC 2025)",
ADDRESS="Bologna, Italy",
PAGES="6.99",
DAYS=1,
MONTH=jul,
YEAR=2025,
KEYWORDS="Serverless; FaaS; Quantization; Edge Intelligence",
ABSTRACT="Edge computing's rise demands efficient compression strategies for
deploying Machine Learning (ML) models on resource-constrained devices. As
Artificial Intelligence (AI) shifts from cloud to edge, optimizing models
across heterogeneous layers is crucial. Quantization reduces numerical
precision, improving model size, inference speed and energy efficiency, key
for edge deployments. However, its complexity limits accessibility. To
address this, we propose Quantization-as-a-Service (QaaS), a serverless
framework that automates model quantization for both cloud and edge
environments. Built on OpenFaaS and Kubernetes, QaaS enables on-demand
execution with dynamic resource orchestration, implementing Layer 5 of Edge
Intelligence (EI). Our evaluation compares quantization performance on edge
devices in terms of CPU usage and execution time when performed as a
service versus locally. Results demonstrate that deploying quantization
workflows using Function-as-a-Service (FaaS) not only maintains
computational efficiency but also reduces CPU consumption compared to
standalone execution, showcasing the potential of serverless solutions in
EI."
}

```
