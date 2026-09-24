{{- define "graph-rag.labels" -}}
app.kubernetes.io/part-of: graph-rag
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "graph-rag.image" -}}
{{ .Values.image.repository }}:{{ required "image.tag is required (sha-<7-char commit>)" .Values.image.tag }}
{{- end }}

{{- define "graph-rag.envFrom" -}}
envFrom:
  - secretRef:
      name: {{ .Values.secretName }}
  - configMapRef:
      name: graph-rag-config
{{- end }}

{{- define "graph-rag.podSecurity" -}}
securityContext:
  runAsNonRoot: true
  runAsUser: 10001
  runAsGroup: 10001
{{- end }}

{{- define "graph-rag.containerSecurity" -}}
securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]
{{- end }}
